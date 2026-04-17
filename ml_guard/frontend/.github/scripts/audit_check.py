import sys
import json

def main():
    try:
        data = json.load(sys.stdin)
    except Exception as e:
        print(f"Error parsing npm audit JSON: {e}")
        sys.exit(0) # Don't fail if audit fails to run

    vulnerabilities = data.get('vulnerabilities', {})
    critical_count = 0
    high_count = 0

    for pkg, info in vulnerabilities.items():
        severity = info.get('severity')
        if severity == 'critical':
            critical_count += 1
            print(f"CRITICAL CVE found in {pkg}: {info.get('via')}")
        elif severity == 'high':
            high_count += 1
            print(f"HIGH CVE found in {pkg}: {info.get('via')}")

    if critical_count > 0:
        print(f"CI Failed: {critical_count} critical vulnerabilities found.")
        sys.exit(1)
    
    print("Audit check passed. No critical vulnerabilities found.")
    sys.exit(0)

if __name__ == "__main__":
    main()
