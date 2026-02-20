import React, { useState, useMemo, useEffect } from 'react'

const Dashboard = ({ token, onLogout }: { token: string, onLogout: () => void }) => {
    const [nlpQuery, setNlpQuery] = useState('')
    const [activeTab, setActiveTab] = useState('scan')
    const [results, setResults] = useState<any>(null)
    const [loading, setLoading] = useState(false)
    const [targetColumn, setTargetColumn] = useState('churn')
    const [error, setError] = useState<string | null>(null)
    const [history, setHistory] = useState<any[]>([])
    const [monitoringData, setMonitoringData] = useState<any[]>([])
    const [projectId] = useState('4a3b2c1d-0e9f-4a3b-b2c1-d0e9f4a3b2c1')

    const API_BASE_URL = `http://${window.location.hostname}:8000/api/v1`

    // File states for Scanning
    const [modelFile, setModelFile] = useState<File | null>(null)
    const [trainFile, setTrainFile] = useState<File | null>(null)
    const [valFile, setValFile] = useState<File | null>(null)

    const nlpOptions = [
        "Run accuracy tests and check for bias",
        "Analyze data quality and check for drift",
        "Perform a comprehensive test suite",
        "Check population stability and performance",
        "Evaluate model robustness and stress test"
    ]

    const stats = useMemo(() => {
        if (!results) return null
        const total = results.results.length
        const passed = results.results.filter((r: any) => r.status === 'passed').length
        const failed = total - passed
        const critical = results.results.filter((r: any) => r.status === 'failed' && r.severity === 'critical').length
        return { total, passed, failed, critical }
    }, [results])

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>, type: string) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0]
            if (type === 'model') setModelFile(file)
            if (type === 'train') setTrainFile(file)
            if (type === 'val') setValFile(file)
            setError(null)
        }
    }

    const fetchHistory = async () => {
        try {
            const resp = await fetch(`${API_BASE_URL}/governance/project/${projectId}/history`, {
                headers: { 'Authorization': `Bearer ${token}` }
            })
            const data = await resp.json()
            setHistory(Array.isArray(data) ? data : [])
        } catch (e) { console.error("History fetch failed", e) }
    }

    const fetchDrift = async () => {
        try {
            const resp = await fetch(`${API_BASE_URL}/governance/project/${projectId}/drift`, {
                headers: { 'Authorization': `Bearer ${token}` }
            })
            const data = await resp.json()
            setMonitoringData(Array.isArray(data) ? data : [])
        } catch (e) { console.error("Drift fetch failed", e) }
    }

    const handleScan = async () => {
        if (!modelFile || !trainFile || !valFile) {
            setError('Missing Artifacts: Please upload Model, Training Set, and Validation Set.')
            return
        }

        setLoading(true)
        setResults(null)
        setError(null)
        try {
            const formData = new FormData()
            formData.append('project_id', projectId)
            formData.append('model_file', modelFile)
            formData.append('train_file', trainFile)
            formData.append('val_file', valFile)
            formData.append('target_column', targetColumn)
            if (nlpQuery) formData.append('query', nlpQuery)

            const response = await fetch(`${API_BASE_URL}/quality-gate/evaluate`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}` },
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json()
                throw new Error(errData.detail || 'Evaluation engine failed')
            }

            const data = await response.json();
            setResults(data)
            fetchHistory()
        } catch (err: any) {
            setError(`Connection Error: ${err.message}`)
        } finally {
            setLoading(false)
        }
    }

    useEffect(() => {
        if (activeTab === 'history') fetchHistory()
        if (activeTab === 'monitoring') fetchDrift()
    }, [activeTab])

    return (
        <div className="min-h-screen bg-[#090A0C] text-slate-200 font-sans selection:bg-orange-500/30">
            {/* Header */}
            <header className="border-b border-white/5 p-5 bg-[#0F1115]/90 backdrop-blur-xl sticky top-0 z-50 flex justify-between items-center shadow-2xl">
                <div className="flex items-center gap-5 group">
                    <div className="w-12 h-12 bg-gradient-to-tr from-orange-600 to-orange-400 rounded-xl flex items-center justify-center font-bold text-2xl text-black shadow-lg shadow-orange-500/40 group-hover:scale-110 transition-all duration-500">
                        🔥
                    </div>
                    <div>
                        <h1 className="text-2xl font-black tracking-tight">ML GUARD <span className="text-orange-500 italic">PRO</span></h1>
                        <p className="text-[10px] text-slate-500 font-bold uppercase tracking-[0.3em] flex items-center gap-2">
                            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                            Enterprise Governance Active
                        </p>
                    </div>
                </div>
                <div className="flex bg-black/40 p-1.5 rounded-2xl border border-white/5 backdrop-blur-md">
                    {['scan', 'train', 'history', 'monitoring'].map((tab) => (
                        <button
                            key={tab}
                            onClick={() => setActiveTab(tab)}
                            className={`px-8 py-2.5 rounded-xl text-xs font-black transition-all duration-300 ${activeTab === tab ? 'bg-orange-500 text-black shadow-[0_0_20px_rgba(249,115,22,0.4)]' : 'hover:text-white text-slate-500'}`}
                        >
                            {tab.replace('_', ' ').toUpperCase()}
                        </button>
                    ))}
                    <button onClick={onLogout} className="px-8 py-2.5 rounded-xl text-xs font-black text-red-500 hover:bg-red-500/10 transition-all">🔒 LOGOUT</button>
                </div>
            </header>

            <main className="p-8 max-w-[1600px] mx-auto">
                {error && (
                    <div className="mb-8 p-4 bg-red-500/10 border border-red-500/20 rounded-2xl flex items-center justify-between text-red-500 text-sm font-bold animate-in slide-in-from-top-4">
                        <div className="flex items-center gap-4"><span className="text-xl">⚠️</span>{error}</div>
                        <button onClick={() => setError(null)} className="text-xs hover:underline uppercase tracking-widest">Dismiss</button>
                    </div>
                )}

                {activeTab === 'scan' && (
                    <div className="grid grid-cols-1 lg:grid-cols-12 gap-10">
                        {/* Control Panel */}
                        <div className="lg:col-span-4 space-y-8">
                            <section className="bg-gradient-to-br from-[#16191E] to-[#0F1115] p-8 rounded-[2.5rem] border border-white/5 shadow-2xl">
                                <h2 className="text-xl font-black mb-6 flex items-center gap-3"><span className="text-orange-500 text-sm">🎙️</span>NLP ANALYZER</h2>
                                <div className="space-y-4 mb-8">
                                    <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Suggested:</p>
                                    <div className="flex flex-wrap gap-2">
                                        {nlpOptions.map((opt, i) => (
                                            <button key={i} onClick={() => setNlpQuery(opt)} className="text-[10px] bg-white/5 border border-white/10 hover:border-orange-500/50 px-3 py-2 rounded-xl text-slate-400 hover:text-white transition-all">{opt}</button>
                                        ))}
                                    </div>
                                </div>
                                <textarea className="w-full bg-black/40 border-2 border-white/5 rounded-2xl p-6 text-sm focus:border-orange-500/50 outline-none h-36 transition-all placeholder:text-slate-700" placeholder="Describe your test intent..." value={nlpQuery} onChange={(e) => setNlpQuery(e.target.value)} />
                                <div className="mt-8 space-y-6">
                                    <div>
                                        <label className="block text-[10px] font-black text-slate-500 uppercase tracking-widest mb-2 px-1">Target Column</label>
                                        <input className="w-full bg-black/40 border-2 border-white/5 rounded-xl p-4 text-sm font-bold focus:border-orange-500/50 outline-none" value={targetColumn} onChange={(e) => setTargetColumn(e.target.value)} />
                                    </div>
                                    <button onClick={handleScan} disabled={loading} className="w-full bg-orange-500 disabled:opacity-50 text-black font-black py-5 rounded-2xl shadow-xl shadow-orange-500/20 uppercase tracking-tighter text-lg">{loading ? 'HYPER-SCANNING...' : 'EXECUTE QUALITY GATE'}</button>
                                </div>
                            </section>

                            <section className="bg-[#0F1115] p-8 rounded-[2.5rem] border border-white/5">
                                <h3 className="text-xs font-black text-slate-500 uppercase tracking-[0.3em] mb-8">Artifact Matrix</h3>
                                <div className="space-y-5">
                                    {[
                                        { label: 'Model Artifact (.pkl)', id: 'model', file: modelFile },
                                        { label: 'Training Data (.csv)', id: 'train', file: trainFile },
                                        { label: 'Validation Data (.csv)', id: 'val', file: valFile }
                                    ].map((item) => (
                                        <div key={item.id} className="group relative">
                                            <input type="file" onChange={(e) => handleFileChange(e, item.id)} className="absolute inset-0 opacity-0 cursor-pointer z-10" />
                                            <div className={`border-2 ${item.file ? 'border-orange-500/50 bg-orange-500/[0.03]' : 'border-white/5 bg-black/20'} rounded-2xl p-5 flex items-center justify-between`}>
                                                <div className="flex flex-col">
                                                    <span className="text-[9px] font-black text-slate-600 uppercase mb-1">{item.label}</span>
                                                    <span className={`text-sm font-black truncate max-w-[180px] ${item.file ? 'text-white' : 'text-slate-700'}`}>{item.file ? item.file.name : 'Not selected'}</span>
                                                </div>
                                                <span className={`${item.file ? 'text-orange-500 rotate-45' : 'text-slate-700'} font-black text-xl`}>＋</span>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </section>
                        </div>

                        {/* Results Panel */}
                        <div className="lg:col-span-8">
                            {results ? (
                                <div className="space-y-8">
                                    <div className={`bg-[#0F1115] border-2 ${results.deployment_allowed ? 'border-green-500/20' : 'border-red-500/20'} p-12 rounded-[3rem] flex justify-between items-center`}>
                                        <div className="space-y-4">
                                            <h2 className={`text-6xl font-black ${results.deployment_allowed ? 'text-white' : 'text-red-500'}`}>{results.deployment_allowed ? 'GATE PASSED' : 'GATE FAIL'}</h2>
                                            <div className="flex gap-4">
                                                <span className="text-[10px] text-slate-500 font-bold uppercase tracking-widest px-3 py-1 bg-white/5 rounded-lg">Tests: {stats?.total}</span>
                                                <span className="text-[10px] text-green-500 font-bold uppercase tracking-widest px-3 py-1 bg-green-500/5 rounded-lg">Pass: {stats?.passed}</span>
                                                <span className="text-[10px] text-red-500 font-bold uppercase tracking-widest px-3 py-1 bg-red-500/5 rounded-lg">Fail: {stats?.failed}</span>
                                                {stats?.critical && stats.critical > 0 ? (
                                                    <span className="text-[10px] text-orange-500 font-black uppercase tracking-widest px-3 py-1 bg-orange-500/10 rounded-lg animate-pulse">Critical: {stats.critical}</span>
                                                ) : null}
                                            </div>
                                        </div>
                                        <div className="text-right">
                                            <div className="text-8xl font-black text-orange-500">{Math.round(results.score)}</div>
                                            <p className="text-[10px] font-black text-slate-500 uppercase mt-4">Quality Index</p>
                                        </div>
                                    </div>
                                    <div className="space-y-4">
                                        {results.results.map((r: any, i: number) => (
                                            <div key={i} className="bg-[#0F1115] p-8 rounded-[2rem] border border-white/5 flex items-center gap-6">
                                                <div className={`w-12 h-12 rounded-xl flex items-center justify-center font-bold ${r.status === 'passed' ? 'bg-green-500/10 text-green-500' : 'bg-red-500/10 text-red-500'}`}>{r.status === 'passed' ? '✓' : '!'}</div>
                                                <div className="flex-1">
                                                    <div className="flex items-center gap-3"><h4 className="font-black text-white">{r.test_name}</h4><span className="text-[8px] bg-slate-800 px-2 py-0.5 rounded uppercase">{r.severity}</span></div>
                                                    <p className="text-xs text-slate-500 mt-1">{r.message}</p>
                                                    {r.explanation && <div className="mt-4 p-4 bg-orange-500/5 border-l-2 border-orange-500 text-[11px] text-slate-300 italic">"{r.explanation}"</div>}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            ) : (
                                <div className="h-[600px] flex flex-col items-center justify-center bg-[#0F1115]/50 border-4 border-dashed border-white/5 rounded-[4rem] text-center p-20">
                                    <div className="text-9xl mb-8 opacity-10">📊</div>
                                    <h3 className="text-4xl font-black text-slate-600 uppercase italic">Awaiting Telemetry</h3>
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {activeTab === 'train' && <TrainerView token={token} />}

                {activeTab === 'history' && (
                    <div className="space-y-8">
                        <h2 className="text-5xl font-black text-white tracking-tighter">PROJECT REGISTRY</h2>
                        <div className="grid grid-cols-1 gap-4">
                            {history.length > 0 ? history.map((run, i) => (
                                <div key={i} className="bg-[#111318] border border-white/5 p-8 rounded-[2rem] flex items-center justify-between">
                                    <div className="flex gap-6 items-center">
                                        <div className={`w-12 h-12 rounded-xl flex items-center justify-center font-bold ${run.deployment_allowed ? 'bg-green-500/10 text-green-500' : 'bg-red-500/10 text-red-500'}`}>{run.deployment_allowed ? '✓' : '✗'}</div>
                                        <div><h4 className="text-white font-black">{run.suite_name}</h4><p className="text-slate-600 text-[10px]">{new Date(run.created_at).toLocaleString()}</p></div>
                                    </div>
                                    <div className="text-right">
                                        <p className="text-slate-600 text-[9px] uppercase font-black">Score</p>
                                        <p className="text-2xl font-black text-orange-500">{Math.round(run.score)}</p>
                                    </div>
                                </div>
                            )) : <div className="text-center py-20 text-slate-600 italic">No history found.</div>}
                        </div>
                    </div>
                )}

                {activeTab === 'monitoring' && (
                    <div className="space-y-12">
                        <h2 className="text-5xl font-black text-white tracking-tighter">DRIFT TELEMETRY</h2>
                        <div className="bg-[#111318] p-12 rounded-[3.5rem] border border-white/5">
                            <div className="h-[300px] flex items-end gap-4">
                                {monitoringData.length > 0 ? monitoringData.map((d, i) => (
                                    <div key={i} className="flex-1 flex flex-col items-center">
                                        <div className={`w-full rounded-t-lg ${d.psi > 0.1 ? 'bg-red-500' : 'bg-orange-500'}`} style={{ height: `${Math.min(d.psi * 1000, 250)}px` }}></div>
                                        <p className="text-[8px] text-slate-600 mt-2 rotate-45 origin-left whitespace-nowrap">{d.feature}</p>
                                    </div>
                                )) : <div className="w-full flex items-center justify-center text-slate-600 italic">Insufficient telemetry data.</div>}
                            </div>
                        </div>
                    </div>
                )}
            </main>
        </div>
    )
}

const TrainerView = ({ token }: { token: string }) => {
    const [dataset, setDataset] = useState<File | null>(null)
    const [target, setTarget] = useState('target')
    const [modelType, setModelType] = useState('random_forest')
    const [loading, setLoading] = useState(false)
    const [trainResult, setTrainResult] = useState<any>(null)
    const [error, setError] = useState<string | null>(null)

    const handleTrain = async () => {
        if (!dataset) return setError('Missing Data')
        setLoading(true)
        try {
            const formData = new FormData()
            formData.append('dataset', dataset)
            formData.append('target_column', target)
            formData.append('model_type', modelType)
            const response = await fetch('http://127.0.0.1:8000/api/v1/quality-gate/train', {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}` },
                body: formData
            })
            const data = await response.json()
            if (!response.ok) throw new Error(data.detail)
            setTrainResult(data)
        } catch (err: any) { setError(err.message) }
        finally { setLoading(false) }
    }

    return (
        <div className="max-w-4xl mx-auto space-y-12">
            <section className="bg-gradient-to-br from-[#16191E] to-[#0D1014] p-16 rounded-[4rem] border border-white/5">
                <h2 className="text-4xl font-black mb-8 text-white uppercase">Training Core</h2>
                <div className="space-y-8">
                    <div className="border-2 border-dashed border-white/5 rounded-3xl p-12 text-center bg-black/40 relative">
                        <input type="file" onChange={(e) => e.target.files && setDataset(e.target.files[0])} className="absolute inset-0 opacity-0 cursor-pointer" />
                        <p className="text-slate-400 font-black">{dataset ? dataset.name : 'Upload CSV Corpus'}</p>
                    </div>
                    <div className="grid grid-cols-2 gap-8">
                        <div><label className="text-[10px] font-black text-slate-500 uppercase">Target</label><input className="w-full bg-black/20 border border-white/5 rounded-xl p-4 text-white" value={target} onChange={(e) => setTarget(e.target.value)} /></div>
                        <div><label className="text-[10px] font-black text-slate-500 uppercase">Type</label><select className="w-full bg-black/20 border border-white/5 rounded-xl p-4 text-white" value={modelType} onChange={(e) => setModelType(e.target.value)}><option value="random_forest">Random Forest</option><option value="gradient_boosting">Gradient Boosting</option></select></div>
                    </div>
                    <button onClick={handleTrain} disabled={loading} className="w-full bg-white text-black font-black py-6 rounded-2xl uppercase tracking-widest hover:bg-orange-500 transition-all">{loading ? 'Training...' : 'Initialize Training'}</button>
                    {trainResult && <div className="p-8 bg-green-500/10 border border-green-500/20 rounded-2xl text-white font-black text-center text-3xl italic">{(trainResult.metrics.accuracy * 100).toFixed(2)}% Accuracy</div>}
                    {error && <div className="text-red-500 font-bold text-center">{error}</div>}
                </div>
            </section>
        </div>
    )
}

export default Dashboard
