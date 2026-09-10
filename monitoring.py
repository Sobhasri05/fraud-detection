import pandas as pd
import numpy as np
from datetime import datetime
import os

print("Starting monitoring...")

# Create reference data (training data)
print("Creating reference data...")
reference_data = pd.DataFrame({
    'amount': np.random.normal(500, 200, 1000),
    'hour': np.random.randint(0, 24, 1000)
})
print(f"Reference data: {len(reference_data)} samples")

# Create current data (production data with drift)
print("Creating current data...")
current_data = pd.DataFrame({
    'amount': np.random.normal(700, 250, 100),  # Higher mean = drift
    'hour': np.random.randint(0, 24, 100)
})
print(f"Current data: {len(current_data)} samples")

# Manual drift calculation
print("\nCalculating drift...")

def calculate_drift(reference, current, feature_name):
    ref_mean = reference[feature_name].mean()
    curr_mean = current[feature_name].mean()
    ref_std = reference[feature_name].std()
    curr_std = current[feature_name].std()
    
    mean_change = abs((curr_mean - ref_mean) / ref_mean) * 100 if ref_mean != 0 else 0
    std_change = abs((curr_std - ref_std) / ref_std) * 100 if ref_std != 0 else 0
    
    drift_detected = mean_change > 10 or std_change > 10
    
    return {
        'feature': feature_name,
        'reference_mean': round(ref_mean, 2),
        'current_mean': round(curr_mean, 2),
        'mean_change_pct': round(mean_change, 2),
        'drift_detected': drift_detected
    }

# Calculate drift for each feature
results = []
for col in reference_data.columns:
    result = calculate_drift(reference_data, current_data, col)
    results.append(result)
    
    status = "DRIFT DETECTED" if result['drift_detected'] else "Stable"
    print(f"\nFeature: {col}")
    print(f"   Reference Mean: {result['reference_mean']}")
    print(f"   Current Mean:   {result['current_mean']}")
    print(f"   Change:         {result['mean_change_pct']}%")
    print(f"   Status:         {status}")

# Generate HTML Report
print("\nGenerating HTML report...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_path = f"drift_report_{timestamp}.html"

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Data Drift Report - {timestamp}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background-color: #3498db; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 12px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .drift-yes {{ color: #e74c3c; font-weight: bold; }}
        .drift-no {{ color: #27ae60; font-weight: bold; }}
        .summary {{ background: #ecf0f1; padding: 20px; border-radius: 5px; margin-top: 20px; }}
        .alert {{ background: #fdecea; border-left: 4px solid #e74c3c; padding: 15px; margin: 20px 0; }}
        .success {{ background: #eafaf1; border-left: 4px solid #27ae60; padding: 15px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Data Drift Monitoring Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Reference Samples:</strong> {len(reference_data)} (Training Data)</p>
        <p><strong>Current Samples:</strong> {len(current_data)} (Production Data)</p>
        
        <h2>Feature Analysis</h2>
        <table>
            <tr>
                <th>Feature</th>
                <th>Reference Mean</th>
                <th>Current Mean</th>
                <th>Change %</th>
                <th>Drift Status</th>
            </tr>
"""

for result in results:
    drift_class = "drift-yes" if result['drift_detected'] else "drift-no"
    drift_text = "YES" if result['drift_detected'] else "NO"
    html_content += f"""
            <tr>
                <td><strong>{result['feature']}</strong></td>
                <td>{result['reference_mean']}</td>
                <td>{result['current_mean']}</td>
                <td>{result['mean_change_pct']}%</td>
                <td class="{drift_class}">{drift_text}</td>
            </tr>
    """

any_drift = any(r['drift_detected'] for r in results)
if any_drift:
    html_content += """
        </table>
        <div class="alert">
            <h3>Data Drift Detected!</h3>
            <p>The following features have drifted significantly from the training data:</p>
            <ul>
    """
    for r in results:
        if r['drift_detected']:
            html_content += f"<li><strong>{r['feature']}</strong> - {r['mean_change_pct']}% change in mean</li>"
    html_content += """
            </ul>
            <p><strong>Action Required:</strong> Consider retraining the model with recent data.</p>
        </div>
    """
else:
    html_content += """
        </table>
        <div class="success">
            <h3>No Significant Drift Detected</h3>
            <p>The model's input data is stable. No action required.</p>
        </div>
    """

html_content += """
        <div class="summary">
            <h3>Summary</h3>
            <p>This report monitors the health of your ML model by comparing production data against training data.</p>
            <p><strong>Drift Threshold:</strong> 10% change in mean or standard deviation</p>
        </div>
        <p style="text-align: center; color: #7f8c8d; margin-top: 30px;">
            Generated by Fraud Detection API Monitoring System
        </p>
    </div>
</body>
</html>
"""

with open(report_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\nReport saved: {report_path}")
print(f"Location: {os.path.abspath(report_path)}")
print("\nMonitoring complete!")
print("Open the HTML file in your browser to see the full report.")