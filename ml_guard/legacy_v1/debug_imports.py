#!/usr/bin/env python3
"""Debug script to test backend imports step by step."""

import sys
import os

def main():
    # Simulate what happens in the Streamlit app
    backend_path = os.path.join(os.getcwd(), 'backend')
    print('Backend path:', backend_path)
    print('Backend exists:', os.path.exists(backend_path))

    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
        print('Added backend to sys.path')

    # Test imports step by step
    try:
        print('Testing structlog import...')
        import structlog
        print('[OK] structlog imported')

        print('Testing pydantic import...')
        import pydantic
        print('[OK] pydantic imported')

        print('Testing fastapi import...')
        import fastapi
        print('[OK] fastapi imported')

        print('Testing config import...')
        from app.core.config import settings
        print('[OK] config imported')

        print('Testing logging setup...')
        from app.core.logging import setup_logging
        setup_logging()
        print('[OK] logging setup')

        print('Testing service imports...')
        from app.services.test_orchestrator import TestOrchestrator
        from app.services.model_registry import ModelRegistry
        print('[OK] services imported')

        print('Testing service instantiation...')
        test_orch = TestOrchestrator()
        model_reg = ModelRegistry()
        print('[OK] services instantiated')

        print('\n[SUCCESS] All backend components working perfectly!')

    except Exception as e:
        print(f'[ERROR] Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()