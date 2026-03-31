import sys
import types
import traceback

sys.path.insert(0, 'ml_guard/backend')
sys.path.insert(0, '.')
# No mocks — let's see real errors
# sys.modules['onnxruntime'] = types.ModuleType('onnxruntime')

import importlib

routers = [
    'advisory','alerts','audit','auth','behavior',
    'ci','datasets','data_quality','deployments',
    'drift','enterprise','experiments','explainability',
    'fairness','forecast','gate','governance','history',
    'ingest','init_scan','jobs','llm_eval','model_registry',
    'monitoring','observe', 'performance','policies','predictions',
    'preflight','red_team','reports','sentinel','streaming'
]

for r in routers:
    try:
        mod = importlib.import_module(f'app.routers.{r}')
        print(f'OK: {r}')
    except Exception as e:
        print(f'FAIL: {r}: {type(e).__name__}: {e}')
        # Log the full traceback to help resolve errors
        # traceback.print_exc()
