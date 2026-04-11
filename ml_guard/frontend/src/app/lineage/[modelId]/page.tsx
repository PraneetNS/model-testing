"use client";

import React, { useEffect, useState, useMemo } from "react";
import { useParams, useRouter } from "next/navigation";
import { ArrowLeft, Activity } from "lucide-react";

interface NodeData {
  id: string;
  name: string;
  version: number;
  score: number | null;
  verdict: string | null;
  created_at: string | null;
}

interface EdgeData {
  parent_id: string;
  child_id: string;
}

interface LineageData {
  nodes: NodeData[];
  edges: EdgeData[];
}

export default function LineagePage() {
  const params = useParams();
  const router = useRouter();
  const modelId = typeof params?.modelId === "string" ? params.modelId : Array.isArray(params?.modelId) ? params.modelId[0] : "";
  const [data, setData] = useState<LineageData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!modelId) return;
    fetch(`http://127.0.0.1:8000/api/lineage/${modelId}/graph`)
      .then((res) => {
        if (!res.ok) throw new Error("Failed to fetch graph");
        return res.json();
      })
      .then((d) => {
        setData(d);
        setLoading(false);
      })
      .catch((e) => {
        setError(e.toString());
        setLoading(false);
      });
  }, [modelId]);

  const { layoutNodes, layoutEdges, width, height } = useMemo(() => {
    if (!data || data.nodes.length === 0) return { layoutNodes: [], layoutEdges: [], width: 800, height: 600 };

    const { nodes, edges } = data;
    const nodeMap = new Map<string, NodeData>();
    nodes.forEach(n => nodeMap.set(n.id, n));

    const childrenMap = new Map<string, string[]>();
    const parentMap = new Map<string, string[]>();

    edges.forEach(e => {
      if (!childrenMap.has(e.parent_id)) childrenMap.set(e.parent_id, []);
      childrenMap.get(e.parent_id)!.push(e.child_id);
      
      if (!parentMap.has(e.child_id)) parentMap.set(e.child_id, []);
      parentMap.get(e.child_id)!.push(e.parent_id);
    });

    const depths = new Map<string, number>();
    const roots = nodes.filter(n => !parentMap.has(n.id) || parentMap.get(n.id)!.length === 0);

    const queue: { id: string; depth: number }[] = roots.map(r => ({ id: r.id, depth: 0 }));
    
    // Assign depth via BFS
    while (queue.length > 0) {
      const { id, depth } = queue.shift()!;
      if (!depths.has(id)) {
        depths.set(id, depth);
        const children = childrenMap.get(id) || [];
        children.forEach(c => queue.push({ id: c, depth: depth + 1 }));
      } else {
        if (depth > depths.get(id)!) {
          depths.set(id, depth); // Update to max depth
          const children = childrenMap.get(id) || [];
          children.forEach(c => queue.push({ id: c, depth: depth + 1 }));
        }
      }
    }

    const layers: { [key: number]: string[] } = {};
    let maxDepth = 0;
    nodes.forEach(n => {
      const d = depths.get(n.id) || 0;
      if (!layers[d]) layers[d] = [];
      layers[d].push(n.id);
      if (d > maxDepth) maxDepth = d;
    });

    const NODE_WIDTH = 220;
    const NODE_HEIGHT = 90;
    const X_SPACING = 300;
    const Y_SPACING = 150;

    const layoutObjNodes: any[] = [];
    let maxCols = 0;

    Object.keys(layers).forEach(depthStr => {
      const depth = parseInt(depthStr);
      const layerNodes = layers[depth];
      if (layerNodes.length > maxCols) maxCols = layerNodes.length;

      const layerWidth = layerNodes.length * X_SPACING;
      
      layerNodes.forEach((nId, idx) => {
        const cx = (idx + 1) * (X_SPACING) - (X_SPACING / 2) - 100; // distribute evenly
        const cy = depth * Y_SPACING + 50;
        layoutObjNodes.push({
          ...nodeMap.get(nId),
          x: cx,
          y: cy,
          width: NODE_WIDTH,
          height: NODE_HEIGHT,
          isTarget: nId === modelId
        });
      });
    });

    const computedWidth = Math.max(800, maxCols * X_SPACING);
    const computedHeight = Math.max(600, (maxDepth + 2) * Y_SPACING);
    
    // Center nodes relative to total width
    layoutObjNodes.forEach(node => {
      const d = depths.get(node.id) || 0;
      const layerSize = layers[d].length;
      const shiftX = (computedWidth - (layerSize * X_SPACING)) / 2;
      node.x += shiftX + 50; 
    });

    const layoutObjEdges = edges.map(e => {
      const parent = layoutObjNodes.find(n => n.id === e.parent_id);
      const child = layoutObjNodes.find(n => n.id === e.child_id);
      return { parent, child };
    }).filter(e => e.parent && e.child);

    return { layoutNodes: layoutObjNodes, layoutEdges: layoutObjEdges, width: computedWidth, height: computedHeight };
  }, [data, modelId]);


  if (loading) return <div className="p-10">Loading DAG...</div>;
  if (error) return <div className="p-10 text-red-500">Error: {error}</div>;

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 p-8">
      <div className="max-w-6xl mx-auto space-y-6">
        <header className="flex items-center gap-4">
          <button onClick={() => router.back()} className="p-2 hover:bg-slate-200 rounded-full transition"><ArrowLeft size={20}/></button>
          <div>
            <h1 className="text-2xl font-bold">Model Lineage DAG</h1>
            <p className="text-sm text-slate-500">Ancestors and Descendants for {modelId}</p>
          </div>
        </header>

        <div className="bg-white border text-[#333] border-slate-200 rounded-xl shadow-sm overflow-auto" style={{ height: "70vh" }}>
          <svg width={width} height={height} className="block mx-auto">
            {/* Defs for arrowheads */}
            <defs>
              <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#94a3b8" />
              </marker>
            </defs>

            {/* Edges */}
            {layoutEdges.map((e, idx) => {
              const startX = e.parent.x + e.parent.width / 2;
              const startY = e.parent.y + e.parent.height;
              const endX = e.child.x + e.child.width / 2;
              const endY = e.child.y;
              
              // Smooth cubic bezier curve
              const pathData = `M ${startX} ${startY} C ${startX} ${startY + 30}, ${endX} ${endY - 30}, ${endX} ${endY}`;
              
              return (
                <path 
                  key={`edge-${idx}`} 
                  d={pathData} 
                  stroke="#cbd5e1" 
                  strokeWidth="2" 
                  fill="none" 
                  markerEnd="url(#arrowhead)" 
                />
              );
            })}

            {/* Nodes */}
            {layoutNodes.map(node => {
              const isCertified = node.verdict === "CERTIFIED";
              const isWarning = node.verdict === "CONDITIONAL";
              const isFailed = node.verdict === "FAILED";
              const isTarget = node.isTarget;

              let bgClass = "#f8fafc";
              let borderClass = "#e2e8f0";
              let titleClass = "#000";

              if (isCertified) { bgClass = "#f0fdf4"; borderClass = "#86efac"; titleClass="#166534"; }
              else if (isWarning) { bgClass = "#fffbeb"; borderClass = "#fcd34d"; titleClass="#92400e"; }
              else if (isFailed) { bgClass = "#fef2f2"; borderClass = "#fca5a5"; titleClass="#991b1b"; }

              return (
                <g key={node.id} transform={`translate(${node.x}, ${node.y})`}>
                  <rect 
                    width={node.width} 
                    height={node.height} 
                    rx="8" 
                    fill={bgClass} 
                    stroke={isTarget ? "#0284c7" : borderClass} 
                    strokeWidth={isTarget ? 3 : 2}
                  />
                  
                  {isTarget && (
                    <text x="-10" y="-10" fontSize="12" fill="#0284c7" fontWeight="bold">CURRENT</text>
                  )}

                  <text x="16" y="24" fontSize="14" fontWeight="600" fill={titleClass} fontFamily="monospace">
                    {node.name} v{node.version}
                  </text>
                  
                  <text x="16" y="44" fontSize="12" fill="#64748b" fontFamily="sans-serif">
                    Score: {node.score !== null ? parseFloat(node.score).toFixed(1) : "N/A"}
                  </text>

                  <text x="16" y="62" fontSize="12" fill="#64748b" fontFamily="sans-serif">
                    Verdict: {node.verdict || "PENDING"}
                  </text>
                  
                  <text x="16" y="80" fontSize="10" fill="#94a3b8" fontFamily="monospace">
                    {node.id.substring(0, 8)}...
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      </div>
    </div>
  );
}
