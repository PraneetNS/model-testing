
import React, { useState, useMemo } from 'react'

const Dashboard = () => {
    const [nlpQuery, setNlpQuery] = useState('')
    const [activeTab, setActiveTab] = useState('scan')
    const [results, setResults] = useState<any>(null)
    const [loading, setLoading] = useState(false)
    const [targetColumn, setTargetColumn] = useState('churn')
    const [error, setError] = useState<string | null>(null)

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
            formData.append('project_id', 'fireflink-enterprise')
            formData.append('model_file', modelFile)
            formData.append('train_file', trainFile)
            formData.append('val_file', valFile)
            formData.append('target_column', targetColumn)
            if (nlpQuery) formData.append('query', nlpQuery)

            const response = await fetch('http://127.0.0.1:8000/api/v1/quality-gate/evaluate', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json()
                throw new Error(errData.detail || 'Evaluation engine failed')
            }

            const data = await response.json();
            setResults(data)
        } catch (err: any) {
            console.error("Scan failed", err);
            setError(`Connection Error: ${err.message}`)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="min-h-screen bg-[#090A0C] text-slate-200 font-sans selection:bg-orange-500/30">
            {/* Header */}
            <header className="border-b border-white/5 p-5 bg-[#0F1115]/90 backdrop-blur-xl sticky top-0 z-50 flex justify-between items-center shadow-2xl">
                <div className="flex items-center gap-5 group">
                    <div className="relative">
                        <div className="w-12 h-12 bg-gradient-to-tr from-orange-600 to-orange-400 rounded-xl flex items-center justify-center font-bold text-2xl text-black shadow-lg shadow-orange-500/40 group-hover:scale-110 transition-all duration-500">
                            🔥
                        </div>
                    </div>
                    <div>
                        <h1 className="text-2xl font-black tracking-tight">
                            ML GUARD <span className="text-orange-500 italic">ENTERPRISE</span>
                        </h1>
                        <p className="text-[10px] text-slate-500 font-bold uppercase tracking-[0.3em] flex items-center gap-2">
                            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                            Restored Fireflink Pipeline Engine
                        </p>
                    </div>
                </div>
                <div className="flex bg-black/40 p-1.5 rounded-2xl border border-white/5 backdrop-blur-md">
                    <button
                        onClick={() => setActiveTab('scan')}
                        className={`px-8 py-2.5 rounded-xl text-xs font-black transition-all duration-300 ${activeTab === 'scan' ? 'bg-orange-500 text-black shadow-[0_0_20px_rgba(249,115,22,0.4)]' : 'hover:text-white text-slate-500'}`}
                    >
                        🚀 QUALITY SCANNER
                    </button>
                    <button
                        onClick={() => setActiveTab('train')}
                        className={`px-8 py-2.5 rounded-xl text-xs font-black transition-all duration-300 ${activeTab === 'train' ? 'bg-orange-500 text-black shadow-[0_0_20px_rgba(249,115,22,0.4)]' : 'hover:text-white text-slate-500'}`}
                    >
                        🧠 MODEL TRAINER
                    </button>
                </div>
            </header>

            <main className="p-8 max-w-[1600px] mx-auto">
                {error && (
                    <div className="mb-8 p-4 bg-red-500/10 border border-red-500/20 rounded-2xl flex items-center gap-4 text-red-500 text-sm font-bold animate-in slide-in-from-top-4">
                        <span className="text-xl">⚠️</span>
                        {error}
                        <button onClick={() => setError(null)} className="ml-auto text-xs hover:underline uppercase tracking-widest">Dismiss</button>
                    </div>
                )}

                {activeTab === 'scan' && (
                    <div className="grid grid-cols-1 lg:grid-cols-12 gap-10">
                        {/* Control Panel */}
                        <div className="lg:col-span-4 space-y-8">
                            <section className="bg-gradient-to-br from-[#16191E] to-[#0F1115] p-8 rounded-[2.5rem] border border-white/5 shadow-2xl relative overflow-hidden group">
                                <h2 className="text-xl font-black mb-6 flex items-center gap-3">
                                    <span className="text-orange-500 text-sm">🎙️</span>
                                    NLP ANALYZER
                                </h2>

                                <div className="space-y-4 mb-8">
                                    <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Suggested:</p>
                                    <div className="flex flex-wrap gap-2">
                                        {nlpOptions.map((opt, i) => (
                                            <button
                                                key={i}
                                                onClick={() => setNlpQuery(opt)}
                                                className="text-[10px] bg-white/5 border border-white/10 hover:border-orange-500/50 px-3 py-2 rounded-xl text-slate-400 hover:text-white transition-all"
                                            >
                                                {opt}
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                <textarea
                                    className="w-full bg-black/40 border-2 border-white/5 rounded-2xl p-6 text-sm focus:border-orange-500/50 outline-none h-36 transition-all placeholder:text-slate-700 font-medium"
                                    placeholder="Describe your test intent..."
                                    value={nlpQuery}
                                    onChange={(e) => setNlpQuery(e.target.value)}
                                />

                                <div className="mt-8 space-y-6">
                                    <div>
                                        <label className="block text-[10px] font-black text-slate-500 uppercase tracking-widest mb-2 px-1">Target Column</label>
                                        <input
                                            className="w-full bg-black/40 border-2 border-white/5 rounded-xl p-4 text-sm font-bold focus:border-orange-500/50 outline-none transition-all"
                                            value={targetColumn}
                                            onChange={(e) => setTargetColumn(e.target.value)}
                                        />
                                    </div>

                                    <button
                                        onClick={handleScan}
                                        disabled={loading}
                                        className="w-full bg-orange-500 disabled:opacity-50 text-black font-black py-5 rounded-2xl transition-all active:scale-[0.98] shadow-xl shadow-orange-500/20 uppercase tracking-tighter text-lg"
                                    >
                                        {loading ? 'HYPER-SCANNING...' : 'EXECUTE QUALITY GATE'}
                                    </button>
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
                                            <input
                                                type="file"
                                                onChange={(e) => handleFileChange(e, item.id)}
                                                className="absolute inset-0 opacity-0 cursor-pointer z-10"
                                            />
                                            <div className={`border-2 ${item.file ? 'border-orange-500/50 bg-orange-500/[0.03]' : 'border-white/5 bg-black/20'} group-hover:border-orange-500/30 rounded-2xl p-5 flex items-center justify-between transition-all`}>
                                                <div className="flex flex-col">
                                                    <span className="text-[9px] font-black text-slate-600 uppercase mb-1">{item.label}</span>
                                                    <span className={`text-sm font-black truncate max-w-[180px] ${item.file ? 'text-white' : 'text-slate-700'}`}>
                                                        {item.file ? item.file.name : 'Not selected'}
                                                    </span>
                                                </div>
                                                <span className={`${item.file ? 'text-orange-500 rotate-45' : 'text-slate-700'} font-black text-xl transition-transform`}>＋</span>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </section>
                        </div>

                        {/* Results Panel */}
                        <div className="lg:col-span-8">
                            {results ? (
                                <div className="space-y-8 animate-in fade-in slide-in-from-bottom-8 duration-1000">
                                    {/* Score Header */}
                                    <div className={`bg-gradient-to-br ${results.deployment_allowed ? 'from-orange-600 to-orange-400' : 'from-red-600 to-red-400'} p-1 rounded-[3rem] shadow-2xl overflow-hidden`}>
                                        <div className="bg-[#0F1115] p-12 rounded-[2.9rem] flex flex-wrap justify-between items-center gap-10">
                                            <div className="space-y-4">
                                                <h2 className={`text-6xl font-black leading-tight tracking-tighter ${results.deployment_allowed ? 'text-white' : 'text-red-500'}`}>
                                                    {results.deployment_allowed ? 'GATE PASSED' : 'GATE FAIL'}
                                                </h2>
                                                <div className="flex gap-4 items-center">
                                                    <span className="bg-white/5 border border-white/10 text-slate-400 px-4 py-1.5 rounded-full text-[10px] font-black tracking-widest uppercase">{results.run_id.split('-').shift()}</span>
                                                    <span className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Status: {results.deployment_allowed ? 'Deployment Recommended' : 'Critical Failure Detected'}</span>
                                                </div>
                                            </div>
                                            <div className="text-right">
                                                <div className={`text-8xl font-black leading-none ${results.deployment_allowed ? 'text-orange-500' : 'text-red-500'}`}>{Math.round(results.score)}</div>
                                                <p className="text-[10px] font-black text-slate-500 tracking-[0.4em] uppercase mt-4">Quality Index</p>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Stats Grid */}
                                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                        {[
                                            { label: 'Total Tests', val: stats?.total, icon: '📋' },
                                            { label: 'Passed', val: stats?.passed, icon: '✅' },
                                            { label: 'Failed', val: stats?.failed, icon: '❌' },
                                            { label: 'Critical', val: stats?.critical, icon: '🔥' }
                                        ].map((s, idx) => (
                                            <div key={idx} className="bg-[#0F1115] border border-white/5 p-6 rounded-3xl">
                                                <div className="text-2xl mb-2">{s.icon}</div>
                                                <div className="text-2xl font-black text-white">{s.val}</div>
                                                <div className="text-[10px] font-black text-slate-500 uppercase tracking-widest mt-1">{s.label}</div>
                                            </div>
                                        ))}
                                    </div>

                                    {/* Test List */}
                                    <div className="space-y-4">
                                        {results.results.map((r: any, i: number) => (
                                            <div key={i} className="group bg-[#0F1115] rounded-[2.5rem] border border-white/5 hover:border-orange-500/20 transition-all duration-500 overflow-hidden">
                                                <div className="p-8 flex flex-wrap justify-between items-center gap-6">
                                                    <div className="flex items-center gap-6">
                                                        <div className={`w-16 h-16 rounded-2xl flex items-center justify-center text-2xl ${r.status === 'passed' ? 'bg-green-500/10 text-green-500' : 'bg-red-500/10 text-red-500'}`}>
                                                            {r.status === 'passed' ? '✓' : '!'}
                                                        </div>
                                                        <div className="flex flex-col gap-1">
                                                            <div className="flex items-center gap-3">
                                                                <h4 className="text-lg font-black text-white tracking-tight">{r.test_name}</h4>
                                                                <span className={`text-[9px] font-black px-2 py-0.5 rounded-lg uppercase tracking-wider ${r.severity === 'critical' ? 'bg-red-500 text-black' : 'bg-slate-800 text-slate-400'}`}>
                                                                    {r.severity}
                                                                </span>
                                                            </div>
                                                            <p className="text-[11px] text-slate-400 font-medium leading-relaxed max-w-2xl">
                                                                {r.description || "Sophisticated statistical validation of model behavioral patterns."}
                                                            </p>

                                                            {r.status === 'failed' && r.explanation && (
                                                                <div id={`remediation-${i}`} className="mt-4 p-4 bg-orange-500/5 border-l-2 border-orange-500 rounded-r-xl">
                                                                    <p className="text-[10px] font-black text-orange-500 uppercase tracking-widest mb-1 flex items-center gap-2">
                                                                        <span>🤖</span> AI REMEDIATION ADVICE
                                                                    </p>
                                                                    <p className="text-[11px] text-slate-300 italic">"{r.explanation}"</p>
                                                                </div>
                                                            )}

                                                            <div className="flex items-center gap-4 mt-2">
                                                                <span className="text-[10px] text-orange-500/80 font-black tracking-widest uppercase">{r.category.replace('_', ' ')}</span>
                                                                {r.actual_value !== undefined && (
                                                                    <span className="text-[10px] text-slate-600 font-mono">
                                                                        Val: <span className={r.status === 'passed' ? 'text-green-500' : 'text-red-500'}>{typeof r.actual_value === 'number' ? r.actual_value.toFixed(4) : r.actual_value}</span>
                                                                        {r.threshold && <span className="opacity-50"> (Target: {r.threshold})</span>}
                                                                    </span>
                                                                )}
                                                            </div>
                                                        </div>
                                                    </div>
                                                    <div className="text-right ml-auto">
                                                        <div className={`text-xs font-black mb-1 px-3 py-1 rounded-full ${r.status === 'passed' ? 'text-green-500 bg-green-500/10' : 'text-red-500 bg-red-500/10'}`}>
                                                            {r.status.toUpperCase()}
                                                        </div>
                                                        <p className="text-[10px] text-slate-600 font-bold">{r.execution_time_seconds.toFixed(3)}s</p>
                                                    </div>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            ) : (
                                <div className="h-[600px] flex flex-col items-center justify-center text-slate-800 space-y-8 bg-[#0F1115]/50 border-4 border-dashed border-white/5 rounded-[4rem] p-20 text-center">
                                    <div className="relative">
                                        <div className="text-[12rem] opacity-[0.03] select-none text-white">📊</div>
                                        <div className="absolute inset-0 flex items-center justify-center">
                                            <div className="w-32 h-32 bg-orange-500/10 rounded-full blur-3xl animate-pulse"></div>
                                        </div>
                                    </div>
                                    <div className="space-y-4 max-w-sm">
                                        <p className="text-4xl font-black tracking-tighter text-slate-600 uppercase italic leading-none">Awaiting Telemetry</p>
                                        <p className="text-[10px] font-bold text-slate-500 leading-relaxed uppercase tracking-[0.2em]">
                                            "Upload artifacts and describe your test intent. Our NLP engine will dynamically select the most relevant test suites for your model."
                                        </p>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {activeTab === 'train' && (
                    <TrainerView />
                )}
            </main>
        </div>
    )
}

const TrainerView = () => {
    const [dataset, setDataset] = useState<File | null>(null)
    const [target, setTarget] = useState('target')
    const [modelType, setModelType] = useState('random_forest')
    const [testSize, setTestSize] = useState(0.2)
    const [doCV, setDoCV] = useState(true)
    const [loading, setLoading] = useState(false)
    const [trainResult, setTrainResult] = useState<any>(null)
    const [error, setError] = useState<string | null>(null)

    const handleTrain = async () => {
        if (!dataset) return setError('Missing Data: Please select a training set.')
        setLoading(true)
        setError(null)
        setTrainResult(null)
        try {
            const formData = new FormData()
            formData.append('dataset', dataset)
            formData.append('target_column', target)
            formData.append('model_type', modelType)
            formData.append('test_size', testSize.toString())
            formData.append('do_cv', doCV.toString())

            const response = await fetch('http://127.0.0.1:8000/api/v1/quality-gate/train', {
                method: 'POST',
                body: formData
            })

            if (!response.ok) {
                const errData = await response.json()
                throw new Error(errData.detail || 'Training engine failed')
            }

            const data = await response.json()
            setTrainResult(data)
        } catch (err: any) {
            setError(`Kernel Error: ${err.message}`)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="max-w-4xl mx-auto space-y-12 animate-in slide-in-from-bottom-10 duration-1000">
            {error && (
                <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-2xl flex items-center gap-4 text-red-500 text-sm font-bold">
                    <span className="text-xl">☣️</span>
                    {error}
                </div>
            )}

            <section className="bg-gradient-to-br from-[#16191E] to-[#0D1014] p-16 rounded-[4rem] border border-white/5 shadow-2xl relative overflow-hidden group">
                <div className="mb-12">
                    <h2 className="text-5xl font-black mb-4 tracking-tighter uppercase leading-none text-white">Restored <br /><span className="text-orange-500">Training Core</span></h2>
                    <p className="text-slate-500 font-medium tracking-tight uppercase text-[10px]">V2.5.0 Production-Ready Engine</p>
                </div>

                <div className="grid grid-cols-1 gap-12">
                    <div>
                        <label className="block text-[10px] font-black text-slate-500 uppercase tracking-widest mb-4 px-2">Primary Training Corpus</label>
                        <div className="group relative border-2 border-dashed border-white/5 rounded-[2.5rem] p-16 text-center hover:border-orange-500/50 transition-all bg-black/40">
                            <input type="file" onChange={(e) => e.target.files && setDataset(e.target.files[0])} className="absolute inset-0 opacity-0 cursor-pointer z-10" />
                            <div className="text-6xl mb-6 opacity-20 group-hover:opacity-100 transition-opacity">📂</div>
                            <p className="text-lg font-black text-slate-400 group-hover:text-white transition-colors">{dataset ? dataset.name : 'Mount Training Data (.csv)'}</p>
                        </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                        <div>
                            <label className="block text-[10px] font-black text-slate-500 mb-3 tracking-widest uppercase px-2">Target Label</label>
                            <input
                                className="w-full bg-black/20 border-2 border-white/5 rounded-2xl p-5 text-sm font-black text-white focus:border-orange-500/50 outline-none transition-all"
                                value={target}
                                onChange={(e) => setTarget(e.target.value)}
                            />
                        </div>
                        <div>
                            <label className="block text-[10px] font-black text-slate-500 mb-3 tracking-widest uppercase px-2">Algorithm</label>
                            <select
                                value={modelType}
                                onChange={(e) => setModelType(e.target.value)}
                                className="w-full bg-black/20 border-2 border-white/5 rounded-2xl p-5 text-sm font-black text-white focus:border-orange-500/50 outline-none appearance-none cursor-pointer"
                            >
                                <option value="random_forest">Random Forest</option>
                                <option value="gradient_boosting">Gradient Boosting</option>
                                <option value="logistic_regression">Logistic Regression</option>
                            </select>
                        </div>
                        <div>
                            <label className="block text-[10px] font-black text-slate-500 mb-3 tracking-widest uppercase px-2">Validation Strategy</label>
                            <div className="flex bg-black/20 p-1.5 rounded-2xl border border-white/5">
                                <button
                                    onClick={() => setDoCV(true)}
                                    className={`flex-1 py-3.5 rounded-xl text-[9px] font-black transition-all ${doCV ? 'bg-orange-500 text-black' : 'text-slate-500'}`}
                                >
                                    5-FOLD CV
                                </button>
                                <button
                                    onClick={() => setDoCV(false)}
                                    className={`flex-1 py-3.5 rounded-xl text-[9px] font-black transition-all ${!doCV ? 'bg-orange-500 text-black' : 'text-slate-500'}`}
                                >
                                    SIMPLE
                                </button>
                            </div>
                        </div>
                    </div>

                    <div>
                        <div className="flex justify-between items-center mb-4 px-2">
                            <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Test Split Ratio</label>
                            <span className="text-[10px] font-black text-orange-500">{(testSize * 100).toFixed(0)}%</span>
                        </div>
                        <input
                            type="range" min="0.1" max="0.5" step="0.05"
                            value={testSize}
                            onChange={(e) => setTestSize(parseFloat(e.target.value))}
                            className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-orange-500"
                        />
                    </div>

                    <button
                        onClick={handleTrain}
                        disabled={loading}
                        className="w-full bg-white text-black font-black py-7 rounded-[2.5rem] hover:bg-orange-500 transition-all active:scale-[0.98] uppercase text-2xl"
                    >
                        {loading ? 'CALCULATING GRADIENTS...' : '🚀 DEPLOY AUTONOMOUS TRAINER'}
                    </button>

                    {trainResult && trainResult.metrics && (
                        <div className="bg-orange-500/5 border border-orange-500/20 rounded-[3rem] p-12 mt-4 relative overflow-hidden animate-in zoom-in-95 duration-700">
                            <div className="flex flex-wrap items-center justify-between gap-12">
                                <div className="space-y-6">
                                    <div className="flex gap-3">
                                        <span className="bg-orange-500 text-black text-[10px] font-black px-4 py-1.5 rounded-full uppercase">OPTIMIZED</span>
                                        <span className="bg-white/5 text-slate-400 text-[10px] font-black px-4 py-1.5 rounded-full uppercase">{trainResult.model_type.replace('_', ' ')}</span>
                                    </div>
                                    <h3 className="text-5xl font-black text-white italic tracking-tighter">
                                        {(trainResult.metrics.accuracy * 100).toFixed(2)}% <span className="text-lg text-slate-600 font-medium not-italic ml-2 uppercase tracking-widest">Accuracy</span>
                                    </h3>

                                    {trainResult.corpus_metadata && (
                                        <div className="flex gap-6 border-y border-white/5 py-3 my-2">
                                            <div className="flex flex-col">
                                                <span className="text-[10px] text-slate-500 font-black uppercase">Corpus Size</span>
                                                <span className="text-xs font-bold text-slate-300">{trainResult.corpus_metadata.rows} Rows</span>
                                            </div>
                                            <div className="flex flex-col">
                                                <span className="text-[10px] text-slate-500 font-black uppercase">Dimensions</span>
                                                <span className="text-xs font-bold text-slate-300">{trainResult.corpus_metadata.columns} Columns</span>
                                            </div>
                                            <div className="flex flex-col">
                                                <span className="text-[10px] text-slate-500 font-black uppercase">Features</span>
                                                <span className="text-xs font-bold text-slate-300">{trainResult.corpus_metadata.features_count} Engineered</span>
                                            </div>
                                        </div>
                                    )}
                                    <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
                                        <div>
                                            <p className="text-slate-600 text-[9px] uppercase font-black mb-1">F1 Score</p>
                                            <p className="text-white text-sm font-black italic">{(trainResult.metrics.f1_score * 100).toFixed(1)}%</p>
                                        </div>
                                        <div>
                                            <p className="text-slate-600 text-[9px] uppercase font-black mb-1">Recall</p>
                                            <p className="text-white text-sm font-black italic">{(trainResult.metrics.recall * 100).toFixed(1)}%</p>
                                        </div>
                                        <div>
                                            <p className="text-slate-600 text-[9px] uppercase font-black mb-1">CV Baseline</p>
                                            <p className="text-white text-sm font-black italic">{(trainResult.metrics.cv_mean * 100).toFixed(1)}%</p>
                                        </div>
                                        <div>
                                            <p className="text-slate-600 text-[9px] uppercase font-black mb-1">Inference</p>
                                            <p className="text-white text-sm font-black italic">8ms</p>
                                        </div>
                                    </div>
                                </div>
                                <div className="flex flex-col gap-4">
                                    <button className="bg-white text-black text-[11px] font-black px-12 py-6 rounded-2xl transition-all uppercase tracking-widest hover:bg-orange-500 hover:text-white shadow-xl shadow-white/5">Export Weights</button>
                                    <p className="text-[9px] text-center text-slate-700 font-bold uppercase tracking-widest">SIG: {trainResult.model_id.split('_').pop()}</p>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </section>
        </div>
    )
}

export default Dashboard
