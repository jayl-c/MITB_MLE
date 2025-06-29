import joblib
import os
import glob
import pandas as pd
# import openpyxl
from datetime import datetime

def get_latest_model(model_type, model_bank_path="model_bank/"):
    """Get the latest model file for a given model type"""
    pattern = os.path.join(model_bank_path, f"*{model_type}*_*.pkl")
    model_files = glob.glob(pattern)
    
    if not model_files:
        print(f"No {model_type} models found in {model_bank_path}")
        return None, None
    
    # Sort by modification time to get the latest
    latest_file = max(model_files, key=os.path.getmtime)
    
    # Load the model package
    model_package = joblib.load(latest_file)
    
    return model_package, latest_file

def generate_model_report():
    """Generate a comprehensive model comparison report"""
    # Load latest models
    xgb_package, xgb_path = get_latest_model("xgb")
    logreg_package, logreg_path = get_latest_model("logreg")
    
    if xgb_package is None or logreg_package is None:
        print("Could not find both model types")
        return
    
    # Extract performance metrics
    xgb_perf = xgb_package['performance']
    logreg_perf = logreg_package['performance']
    
    # Create comparison DataFrame
    comparison_data = {
        'Metric': ['Train AUC', 'Test AUC', 'OOT AUC', 'Best Threshold', 
                   'Max F-beta Train', 'Max F-beta Test', 'Max F-beta OOT'],
        'XGBoost': [
            xgb_perf['train_auc'],
            xgb_perf['test_auc'], 
            xgb_perf['oot_auc'],
            xgb_perf['best_threshold'],
            xgb_perf['max_fbeta_train'],
            xgb_perf['max_fbeta_test'],
            xgb_perf['max_fbeta_oot']
        ],
        'Logistic Regression': [
            logreg_perf['train_auc'],
            logreg_perf['test_auc'],
            logreg_perf['oot_auc'], 
            logreg_perf['best_threshold'],
            logreg_perf['max_fbeta_train'],
            logreg_perf['max_fbeta_test'],
            logreg_perf['max_fbeta_oot']
        ]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison['Difference (XGB - LogReg)'] = (
        df_comparison['XGBoost'] - df_comparison['Logistic Regression']
    )
    
    # Calculate overfitting metrics
    xgb_overfit = xgb_perf['train_auc'] - xgb_perf['oot_auc']
    logreg_overfit = logreg_perf['train_auc'] - logreg_perf['oot_auc']
    
    # Generate timestamp for report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create comprehensive report data
    report_data = []
    
    # Add performance comparison
    for _, row in df_comparison.iterrows():
        report_data.append({
            'Section': 'Performance',
            'Metric': row['Metric'],
            'XGBoost': row['XGBoost'],
            'Logistic_Regression': row['Logistic Regression'],
            'Difference_XGB_minus_LogReg': row['Difference (XGB - LogReg)'],
            'Notes': ''
        })
    
    # Add overfitting analysis
    report_data.append({
        'Section': 'Overfitting',
        'Metric': 'Train-OOT AUC Gap',
        'XGBoost': xgb_overfit,
        'Logistic_Regression': logreg_overfit,
        'Difference_XGB_minus_LogReg': xgb_overfit - logreg_overfit,
        'Notes': 'Higher gap indicates more overfitting'
    })
    
    # Add model metadata
    report_data.append({
        'Section': 'Model Files',
        'Metric': 'XGBoost File',
        'XGBoost': os.path.basename(xgb_path),
        'Logistic_Regression': '',
        'Difference_XGB_minus_LogReg': '',
        'Notes': ''
    })
    
    report_data.append({
        'Section': 'Model Files',
        'Metric': 'LogReg File',
        'XGBoost': '',
        'Logistic_Regression': os.path.basename(logreg_path),
        'Difference_XGB_minus_LogReg': '',
        'Notes': ''
    })
    
    # Convert to DataFrame
    report_df = pd.DataFrame(report_data)
    
    # Create separate hyperparameters DataFrame
    xgb_params_df = pd.DataFrame([
        {'Model': 'XGBoost', 'Parameter': k, 'Value': v} 
        for k, v in xgb_package['best_params'].items()
    ])
    
    logreg_params_df = pd.DataFrame([
        {'Model': 'Logistic_Regression', 'Parameter': k, 'Value': v} 
        for k, v in logreg_package['best_params'].items()
    ])
    
    params_df = pd.concat([xgb_params_df, logreg_params_df], ignore_index=True)
    
    # Save to Excel with multiple sheets
    os.makedirs("reports", exist_ok=True)
    excel_filename = f"reports/model_comparison_report_{timestamp}.xlsx"
    
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        report_df.to_excel(writer, sheet_name='Comparison', index=False)
        params_df.to_excel(writer, sheet_name='Hyperparameters', index=False)
        df_comparison.to_excel(writer, sheet_name='Performance_Only', index=False)
    
    # Also save simple CSV
    csv_filename = f"reports/model_comparison_{timestamp}.csv"
    df_comparison.to_csv(csv_filename, index=False)
    
    print(f"Excel Report (multi-sheet) saved to: {excel_filename}")
    print(f"Simple CSV saved to: {csv_filename}")
    print(f"\nOpen the Excel file to see:")
    print("- 'Comparison' sheet: Full report with sections")
    print("- 'Hyperparameters' sheet: Best parameters for both models")
    print("- 'Performance_Only' sheet: Just the performance metrics")
    
    return df_comparison, excel_filename

# Run the report generation
if __name__ == "__main__":
    comparison_df, report_path = generate_model_report()

