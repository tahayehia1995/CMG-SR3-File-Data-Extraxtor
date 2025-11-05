#!/usr/bin/env python3
"""
Quick test script to validate app imports and basic functionality
"""

import sys
import os

print("Testing SR3 Web Application...")
print("=" * 50)

# Test 1: Import streamlit
try:
    import streamlit as st
    print(f"✅ Streamlit imported (version: {st.__version__})")
except ImportError as e:
    print(f"❌ Failed to import streamlit: {e}")
    sys.exit(1)

# Test 2: Import wrapper modules
try:
    from streamlit_extractor import StreamlitSR3Extractor
    print("✅ StreamlitSR3Extractor imported")
except Exception as e:
    print(f"❌ Failed to import StreamlitSR3Extractor: {e}")
    sys.exit(1)

try:
    from streamlit_visualizer import StreamlitH5Visualizer
    print("✅ StreamlitH5Visualizer imported")
except Exception as e:
    print(f"❌ Failed to import StreamlitH5Visualizer: {e}")
    sys.exit(1)

# Test 3: Import core modules
try:
    from interactive_sr3_extractor import BatchSR3Extractor
    print("✅ BatchSR3Extractor imported")
except Exception as e:
    print(f"❌ Failed to import BatchSR3Extractor: {e}")
    sys.exit(1)

try:
    from interactive_h5_visualizer import InteractiveH5Visualizer
    print("✅ InteractiveH5Visualizer imported")
except Exception as e:
    print(f"❌ Failed to import InteractiveH5Visualizer: {e}")
    sys.exit(1)

# Test 4: Check app.py syntax
try:
    with open('app.py', 'r', encoding='utf-8') as f:
        code = f.read()
    compile(code, 'app.py', 'exec')
    print("✅ app.py syntax is valid")
except SyntaxError as e:
    print(f"❌ app.py has syntax errors: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error checking app.py: {e}")
    sys.exit(1)

# Test 5: Check run_app.py syntax
try:
    with open('run_app.py', 'r', encoding='utf-8') as f:
        code = f.read()
    compile(code, 'run_app.py', 'exec')
    print("✅ run_app.py syntax is valid")
except SyntaxError as e:
    print(f"❌ run_app.py has syntax errors: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error checking run_app.py: {e}")
    sys.exit(1)

print("=" * 50)
print("✅ All tests passed! The application is ready to run.")
print("\nTo start the app, run:")
print("  python run_app.py")
print("\nOr for network access:")
print("  python run_app.py --host 0.0.0.0")

