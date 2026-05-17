import type { Metadata } from 'next';
import Link from 'next/link';
import { CodeBlock } from '@/components/ui/CodeBlock';
import { ChevronRight } from 'lucide-react';

const DOC_CONTENT: Record<string, { title: string; body: React.ReactNode }> = {
  'quick-start': {
    title: 'Quick Start',
    body: (
      <div className="prose-doc">
        <p>Audit your first model in under 5 minutes.</p>
        <h2>1. Install the SDK</h2>
        <CodeBlock code="pip install niyantrana" language="bash" />
        <h2>2. Authenticate</h2>
        <CodeBlock code={`export NIYANTRANA_API_KEY="niy_your_key_here"`} language="bash" />
        <h2>3. Run your first audit</h2>
        <CodeBlock
          code={`from niyantrana import NiyantranaClient\n\nclient = NiyantranaClient()\nresult = client.audit_model(\n    model_path="./my_model.pkl",\n    test_data="./test.csv",\n    contracts=["fairness_parity", "confidence_threshold"]\n)\n\nprint(result.governance_score)  # 91.4\nprint(result.verdict)           # CERTIFIED\nresult.download_certificate("./cert.pdf")`}
          language="python"
        />
        <h2>4. Check the dashboard</h2>
        <p>Log in to <Link href="/dashboard" className="text-forest underline underline-offset-4">your dashboard</Link> to view the full report, drift trends, and compliance mappings.</p>
      </div>
    ),
  },
  'behavioral-contracts': {
    title: 'Behavioral Contracts',
    body: (
      <div className="prose-doc">
        <p>A Behavioral Contract is a machine-readable promise your model must keep. Every prediction is validated against it in real time.</p>
        <h2>Defining a contract</h2>
        <CodeBlock
          code={`contracts:\n  - name: confidence_threshold\n    type: threshold\n    metric: prediction_confidence\n    operator: ">="  \n    value: 0.85\n    on_breach: log_and_alert\n\n  - name: fairness_parity\n    type: fairness\n    metric: demographic_parity\n    groups: [age_group, gender]\n    max_disparity: 0.05\n    on_breach: block`}
          language="python"
        />
        <h2>Breach handling</h2>
        <p>When a contract is breached, Niyantrana can: <code className="text-code">log</code>, <code className="text-code">alert</code>, or <code className="text-code">block</code> the prediction. Each breach is recorded with a cryptographic timestamp.</p>
      </div>
    ),
  },
  'aibom': {
    title: 'AIBOM — AI Bill of Materials',
    body: (
      <div className="prose-doc">
        <p>An AIBOM is a structured inventory of everything your model depends on, required under EU AI Act Article 13.</p>
        <h2>Generate an AIBOM</h2>
        <CodeBlock
          code={`niyantrana aibom generate --model ./my_model.pkl --output aibom.json`}
          language="bash"
        />
        <h2>AIBOM contents</h2>
        <ul>
          <li>Model weights SHA-256 hash</li>
          <li>Training dataset provenance and hashes</li>
          <li>All Python dependencies with CVE status</li>
          <li>Framework versions (scikit-learn, PyTorch, etc.)</li>
        </ul>
      </div>
    ),
  },
  'installation': {
    title: 'Installation',
    body: (
      <div className="prose-doc">
        <p>Install the Niyantrana SDK to start auditing your ML models.</p>
        <h2>Using Pip</h2>
        <CodeBlock code="pip install niyantrana" language="bash" />
        <h2>Using Poetry</h2>
        <CodeBlock code="poetry add niyantrana" language="bash" />
        <h2>Requirements</h2>
        <ul>
          <li>Python 3.8+</li>
          <li>scikit-learn, PyTorch, or TensorFlow</li>
          <li>pandas and numpy</li>
        </ul>
      </div>
    ),
  },
  'authentication': {
    title: 'Authentication',
    body: (
      <div className="prose-doc">
        <p>To use the Niyantrana API and SDK, you need an API key.</p>
        <h2>Generating an API Key</h2>
        <ol className="list-decimal ml-4 mb-4">
          <li>Log in to your Niyantrana dashboard.</li>
          <li>Navigate to <b>Settings &gt; API Keys</b>.</li>
          <li>Click <b>Generate New Key</b> and copy the token.</li>
        </ol>
        <h2>Setting Environment Variables</h2>
        <CodeBlock code={`export NIYANTRANA_API_KEY="niy_your_key_here"`} language="bash" />
        <p>The SDK will automatically pick up this environment variable.</p>
      </div>
    ),
  },
  'governance-scoring': {
    title: 'Governance Scoring',
    body: (
      <div className="prose-doc">
        <p>The Governance Score is a 0–100 metric reflecting a model's compliance, fairness, robustness, and explainability.</p>
        <h2>How it works</h2>
        <ul>
          <li><b>Security & Vulnerability (30%)</b>: Based on CVE scans and adversarial testing.</li>
          <li><b>Fairness & Bias (30%)</b>: Demographic parity and equalized odds.</li>
          <li><b>Explainability (20%)</b>: SHAP values concentration and interpretability.</li>
          <li><b>Behavioral Compliance (20%)</b>: Adherence to behavioral contracts.</li>
        </ul>
        <h2>Passing Threshold</h2>
        <p>By default, models require a score of <b>80 or higher</b> to pass the CI/CD deployment gate.</p>
      </div>
    ),
  },
  'drift-detection': {
    title: 'Drift Detection',
    body: (
      <div className="prose-doc">
        <p>Monitor your deployed models for data and concept drift to prevent silent failures.</p>
        <h2>Supported Metrics</h2>
        <ul>
          <li><b>Population Stability Index (PSI)</b>: Measures distributional shifts in numerical features.</li>
          <li><b>Kolmogorov-Smirnov (KS) Test</b>: Checks if two samples are drawn from the same distribution.</li>
          <li><b>Jensen-Shannon Divergence</b>: Measures the similarity between two probability distributions.</li>
        </ul>
        <h2>Setting up a Drift Monitor</h2>
        <CodeBlock
          code={`from niyantrana.monitoring import DriftMonitor\n\nmonitor = DriftMonitor(model_id="customer_churn_v1")\nmonitor.log_prediction(features=data, prediction=pred)\n\nstatus = monitor.check_drift()\nprint(status.is_drifting) # True`}
          language="python"
        />
      </div>
    ),
  },
  'api-reference': {
    title: 'API Reference',
    body: (
      <div className="prose-doc">
        <p>Integrate Niyantrana directly into your custom tools via our REST API.</p>
        <h2>Base URL</h2>
        <CodeBlock code="https://api.niyantrana.ai/v1" language="bash" />
        <h2>Authentication</h2>
        <p>Pass your API key in the <code className="text-code">X-API-Key</code> header.</p>
        <h2>Common Endpoints</h2>
        <ul>
          <li><code className="text-code">GET /models</code> - List registered models</li>
          <li><code className="text-code">POST /scans/audit</code> - Trigger a model audit</li>
          <li><code className="text-code">GET /contracts/&#123;model_id&#125;</code> - Fetch behavioral contracts</li>
        </ul>
        <p>For full OpenAPI documentation, visit <Link href="/docs/swagger" className="text-forest underline">api.niyantrana.ai/docs</Link>.</p>
      </div>
    ),
  },
  'cicd-setup': {
    title: 'CI/CD Setup',
    body: (
      <div className="prose-doc">
        <p>Block non-compliant models from being deployed by integrating Niyantrana into your CI/CD pipelines.</p>
        <h2>GitHub Actions Example</h2>
        <CodeBlock
          code={`name: ML Governance Gate\non: [push]\n\njobs:\n  audit:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v3\n      - name: Install Niyantrana\n        run: pip install niyantrana\n      - name: Run Audit\n        env:\n          NIYANTRANA_API_KEY: \${{ secrets.NIYANTRANA_API_KEY }}\n        run: niyantrana ci audit ./model.pkl`}
          language="yaml"
        />
        <p>If the governance score falls below the required threshold, the build will fail.</p>
      </div>
    ),
  },
  'huggingface': {
    title: 'HuggingFace Integration',
    body: (
      <div className="prose-doc">
        <p>Audit and scan models directly from the HuggingFace Hub.</p>
        <h2>Auditing a Hub Model</h2>
        <CodeBlock
          code={`from niyantrana.integrations import HuggingFace\n\n# Scans weights, AIBOM, and license compliance\nreport = HuggingFace.audit("distilbert-base-uncased")\nprint(report.verdict)`}
          language="python"
        />
        <p>This automatically downloads the model, runs static analysis, and generates an AIBOM mapping the transformers library dependencies.</p>
      </div>
    ),
  },
  'mlflow': {
    title: 'MLflow / W&B Integration',
    body: (
      <div className="prose-doc">
        <p>Sync your MLflow and Weights & Biases experiment runs with Niyantrana.</p>
        <h2>MLflow Auto-logging</h2>
        <CodeBlock
          code={`import mlflow\nimport niyantrana.integrations.mlflow as n_mlflow\n\nn_mlflow.autolog()\n\nwith mlflow.start_run():\n    # Train your model\n    model.fit(X_train, y_train)\n    # Niyantrana automatically captures the model and generates a governance score`}
          language="python"
        />
        <p>The governance score will be logged as an MLflow metric.</p>
      </div>
    ),
  },
  'slack-teams': {
    title: 'Slack / Teams Alerts',
    body: (
      <div className="prose-doc">
        <p>Get instant notifications when a model breaches a behavioral contract or experiences drift.</p>
        <h2>Configuring Webhooks</h2>
        <ol className="list-decimal ml-4 mb-4">
          <li>Go to <b>Settings &gt; Integrations</b> in the dashboard.</li>
          <li>Paste your Slack Incoming Webhook URL.</li>
          <li>Select the severity level (e.g., Critical and High).</li>
        </ol>
        <p>Alerts will include a direct link to the breach report and the specific metric that failed.</p>
      </div>
    ),
  },
};

const DEFAULT_DOC = {
  title: 'Documentation',
  body: <p>This documentation page is coming soon. <Link href="/docs" className="text-forest underline underline-offset-4">Return to docs index</Link>.</p>,
};

const NAV = [
  { slug: 'quick-start', label: 'Quick start' },
  { slug: 'installation', label: 'Installation' },
  { slug: 'authentication', label: 'Authentication' },
  { slug: 'behavioral-contracts', label: 'Behavioral Contracts' },
  { slug: 'governance-scoring', label: 'Governance Scoring' },
  { slug: 'drift-detection', label: 'Drift Detection' },
  { slug: 'aibom', label: 'AIBOM' },
  { slug: 'api-reference', label: 'API Reference' },
  { slug: 'cicd-setup', label: 'CI/CD Setup' },
  { slug: 'huggingface', label: 'HuggingFace' },
  { slug: 'mlflow', label: 'MLflow / W&B' },
  { slug: 'slack-teams', label: 'Slack / Teams' },
];

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const { slug } = await params;
  const doc = DOC_CONTENT[slug] ?? DEFAULT_DOC;
  return {
    title: `${doc.title} — Niyantrana Docs`,
    description: `Niyantrana documentation: ${doc.title}`,
  };
}

export default async function DocSlugPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const doc = DOC_CONTENT[slug] ?? DEFAULT_DOC;

  return (
    <div className="flex min-h-screen">
      {/* Left sidebar */}
      <aside className="hidden lg:block w-[240px] flex-shrink-0 border-r border-stone bg-white">
        <div className="sticky top-16 p-5 overflow-y-auto max-h-[calc(100vh-64px)]">
          <p className="text-[10px] font-bold uppercase tracking-[0.08em] text-muted mb-3">Documentation</p>
          <nav className="flex flex-col gap-0.5">
            {NAV.map((item) => (
              <Link
                key={item.slug}
                href={`/docs/${item.slug}`}
                className={`px-3 py-2 text-[13px] rounded-[6px] transition-colors duration-150 ${
                  item.slug === slug
                    ? 'bg-mist text-forest font-medium'
                    : 'text-ink-soft hover:text-ink hover:bg-ivory'
                }`}
              >
                {item.label}
              </Link>
            ))}
          </nav>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 px-8 py-12 max-w-[760px]">
        {/* Breadcrumb */}
        <div className="flex items-center gap-1 text-[12px] text-muted mb-8">
          <Link href="/docs" className="hover:text-ink transition-colors duration-150">Docs</Link>
          <ChevronRight size={12} strokeWidth={1.5} />
          <span className="text-ink-soft">{doc.title}</span>
        </div>

        <h1 className="text-3xl font-bold text-ink mb-6" style={{ letterSpacing: '-0.02em' }}>
          {doc.title}
        </h1>

        <div className="text-[15px] text-ink-soft leading-relaxed [&_h2]:text-xl [&_h2]:font-semibold [&_h2]:text-ink [&_h2]:mt-8 [&_h2]:mb-3 [&_h2]:tracking-[-0.01em] [&_p]:mb-4 [&_ul]:mb-4 [&_ul]:pl-4 [&_li]:mb-1.5 [&_li]:list-disc">
          {doc.body}
        </div>
      </main>
    </div>
  );
}
