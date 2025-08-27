#!/usr/bin/env python3
"""
Verification script to check if resource cleanup has been properly implemented
"""

import os
import re

def check_file_cleanup(filepath, filename):
    """Check if a file has proper cleanup implementation"""
    print(f"\nChecking {filename}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    checks = {
        'signal_imports': bool(re.search(r'import signal', content)),
        'signal_handler': bool(re.search(r'def signal_handler.*:', content)),
        'signal_setup': bool(re.search(r'signal\.signal\(signal\.SIG', content)),
        'try_block': bool(re.search(r'\s+try:', content)),
        'finally_block': bool(re.search(r'\s+finally:', content)),
        'cleanup_logging': bool(re.search(r'(Starting cleanup|cleanup starting)', content, re.IGNORECASE)),
        'env_close': bool(re.search(r'env\.close\(\)', content)),
        'error_handling': bool(re.search(r'except.*Exception.*:', content)),
    }
    
    # Print results
    print(f"  ✓ Signal handling imports: {'✅' if checks['signal_imports'] else '❌'}")
    print(f"  ✓ Signal handler function: {'✅' if checks['signal_handler'] else '❌'}")
    print(f"  ✓ Signal setup (SIGINT/SIGTERM): {'✅' if checks['signal_setup'] else '❌'}")
    print(f"  ✓ Try-Finally structure: {'✅' if checks['try_block'] and checks['finally_block'] else '❌'}")
    print(f"  ✓ Cleanup logging: {'✅' if checks['cleanup_logging'] else '❌'}")
    print(f"  ✓ Environment close: {'✅' if checks['env_close'] else '❌'}")
    print(f"  ✓ Exception handling: {'✅' if checks['error_handling'] else '❌'}")
    
    # Overall score
    score = sum(checks.values())
    total = len(checks)
    print(f"  Overall: {score}/{total} checks passed")
    
    return score == total

def check_train_rlpd_specific(filepath):
    """Additional checks specific to train_rlpd.py"""
    print("\n  Additional checks for train_rlpd.py:")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    checks = {
        'client_stop': bool(re.search(r'client\.stop\(\)', content)),
        'server_stop': bool(re.search(r'server\.stop\(\)', content)),
        'jax_cleanup': bool(re.search(r'jax\.clear_caches\(\)', content)),
        'gc_collect': bool(re.search(r'gc\.collect\(\)', content)),
        'pbar_close': bool(re.search(r'pbar\.close\(\)', content)),
    }
    
    print(f"    ✓ TrainerClient stop: {'✅' if checks['client_stop'] else '❌'}")
    print(f"    ✓ TrainerServer stop: {'✅' if checks['server_stop'] else '❌'}")
    print(f"    ✓ JAX memory cleanup: {'✅' if checks['jax_cleanup'] else '❌'}")
    print(f"    ✓ Garbage collection: {'✅' if checks['gc_collect'] else '❌'}")
    print(f"    ✓ Progress bar cleanup: {'✅' if checks['pbar_close'] else '❌'}")
    
    return all(checks.values())

def main():
    print("=" * 60)
    print("Resource Cleanup Implementation Verification")
    print("=" * 60)
    
    base_path = "/home/hanyu/code/hil-serl/examples"
    files_to_check = [
        ("record_success_fail.py", os.path.join(base_path, "record_success_fail.py")),
        ("record_demos.py", os.path.join(base_path, "record_demos.py")),
        ("train_rlpd.py", os.path.join(base_path, "train_rlpd.py")),
    ]
    
    all_passed = True
    
    for filename, filepath in files_to_check:
        if not os.path.exists(filepath):
            print(f"\n❌ {filename} not found at {filepath}")
            all_passed = False
            continue
        
        passed = check_file_cleanup(filepath, filename)
        
        # Additional checks for train_rlpd.py
        if filename == "train_rlpd.py":
            train_passed = check_train_rlpd_specific(filepath)
            passed = passed and train_passed
        
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All resource cleanup implementations verified successfully!")
        print("\nKey improvements:")
        print("- Signal handling (SIGINT/SIGTERM) for graceful shutdown")
        print("- Try-finally blocks ensuring cleanup even on errors")
        print("- Individual resource cleanup with error handling")
        print("- JAX memory cleanup for GPU resources")
        print("- TrainerServer/Client proper shutdown")
        print("- Clear logging of cleanup progress")
    else:
        print("❌ Some checks failed. Please review the implementation.")
    print("=" * 60)
    
    print("\n💡 Tips to prevent GPU memory issues:")
    print("1. Always run with: export XLA_PYTHON_CLIENT_PREALLOCATE=false")
    print("2. For actor: export XLA_PYTHON_CLIENT_MEM_FRACTION=.1")
    print("3. For learner: export XLA_PYTHON_CLIENT_MEM_FRACTION=.3")
    print("4. Use the quick_clear_gpu.sh script if processes get stuck")

if __name__ == "__main__":
    main()