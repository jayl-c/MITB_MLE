# Enhanced XGBoost Training with F1 Metrics and Smart Registration

import mlflow
import mlflow.xgboost
from sklearn.metrics import roc_auc_score, fbeta_score
import numpy as np

def calculate_best_threshold_f1(y_true, y_pred_proba, beta=1.0):
    """Calculate best threshold based on F-beta score"""
    thresholds = np.arange(0.01, 1.0, 0.01)
    f_scores = [fbeta_score(y_true, (y_pred_proba > t).astype(int), beta=beta, zero_division=0) 
                for t in thresholds]
    best_idx = np.argmax(f_scores)
    return thresholds[best_idx], f_scores[best_idx]

# Set up MLflow
mlflow.set_tracking_uri(uri="http://mlflow:5000")
mlflow.set_experiment("Loan Default Prediction")

with mlflow.start_run(run_name=f"xgb_{model_train_date_str}") as run:
    
    # Set up standard scaler preprocessing
    scaler = StandardScaler()
    transformer_stdscaler = scaler.fit(X_train)
    
    with open("standard_scaler.pkl", "wb") as f:
        pickle.dump(transformer_stdscaler, f)
    mlflow.log_artifact("standard_scaler.pkl")
    
    # Transform data
    X_train_processed = transformer_stdscaler.transform(X_train)
    X_test_processed = transformer_stdscaler.transform(X_test)
    X_oot_processed = transformer_stdscaler.transform(X_oot)
    
    print('X_train_processed', X_train_processed.shape[0])
    print('X_test_processed', X_test_processed.shape[0])
    print('X_oot_processed', X_oot_processed.shape[0])
    
    run_id = run.info.run_id
    
    # Log parameters for traceability
    mlflow.log_param("model_type", "XGBoostClassifier")
    mlflow.log_param("feature_selection", "top_20_importance_cutoff_0.01")
    mlflow.log_param("snapshot_date", config["train_test_start_date"])
    
    # Train initial model for feature selection
    print("Performing feature selection...")
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=10,
        learning_rate=0.1,
        use_label_encoder=False,
        objective="binary:logistic",
        random_state=42,
    )
    xgb_model.fit(X_train_processed, y_train)
    
    # Extract feature importances
    importance = xgb_model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importance
    }).sort_values(by="importance", ascending=False)
    
    print(importance_df.head(20))
    
    # Create and save feature importance plot
    output_dir = "output/plots"
    os.makedirs(output_dir, exist_ok=True)
    plot_filepath = os.path.join(output_dir, "top_20_feature_importance.png")
    
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df["feature"][:20], importance_df["importance"][:20])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 20 Feature Importances")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(plot_filepath, dpi=300)
    mlflow.log_artifact(plot_filepath)
    print(f"Feature importance plot saved to: {plot_filepath}")
    
    # Select top features
    importance_cutoff = 0.01
    top_features_df = importance_df.head(20)
    selected_features = top_features_df[top_features_df["importance"] >= importance_cutoff]["feature"].tolist()
    print("Selected features:", selected_features)
    
    # Save selected features
    with open("selected_features.json", "w") as f:
        json.dump(selected_features, f)
    mlflow.log_artifact("selected_features.json")
    
    # Create feature index mapping and get feature subset
    feature_index_map = {feature: idx for idx, feature in enumerate(selected_features)}
    selected_indices = [feature_cols.index(feat) for feat in selected_features]
    
    X_train_cut = X_train_processed[:, selected_indices]
    X_test_cut = X_test_processed[:, selected_indices]
    X_oot_cut = X_oot_processed[:, selected_indices]
    
    # XGBOOST HYPERPARAMETER TUNING
    
    print("Performing XGBoost hyperparameter tuning...")
    xgb_clf = xgb.XGBClassifier(objective="binary:logistic", random_state=88)
    
    # Define hyperparameter space
    param_dist = {
        'n_estimators': [25, 50],
        'max_depth': [2, 3],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.6, 0.8],
        'gamma': [0, 0.1],
        'min_child_weight': [1, 3, 5],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 1.5, 2]
    }
    
    # Create AUC scorer
    auc_scorer = make_scorer(roc_auc_score)
    
    # Set up random search with cross-validation
    random_search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_dist,
        scoring=auc_scorer,
        n_iter=100,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    # Perform hyperparameter search
    random_search.fit(X_train_cut, y_train)
    
    # Log best hyperparameters
    mlflow.log_params(random_search.best_params_)
    best_model = random_search.best_estimator_
    
    # MODEL EVALUATION WITH ENHANCED METRICS
    
    print("\nEvaluating optimized XGBoost model...")
    
    metric_dict = {}
    threshold_dict = {}
    
    for dataset_name, X_data, y_data in [
        ("train", X_train_cut, y_train),
        ("test", X_test_cut, y_test),
        ("oot", X_oot_cut, y_oot),
    ]:
        # Get prediction probabilities
        y_pred_proba = best_model.predict_proba(X_data)[:, 1]
        
        # Calculate AUC
        auc = roc_auc_score(y_data, y_pred_proba)
        metric_dict[f"{dataset_name}_auc"] = auc
        
        # Calculate best F1 threshold and score
        best_threshold, best_f1 = calculate_best_threshold_f1(y_data, y_pred_proba)
        metric_dict[f"{dataset_name}_f1"] = best_f1
        threshold_dict[f"{dataset_name}_threshold"] = best_threshold
        
        # Log metrics to MLflow
        mlflow.log_metric(f"{dataset_name}_auc", auc)
        mlflow.log_metric(f"{dataset_name}_f1", best_f1)
        mlflow.log_metric(f"{dataset_name}_threshold", best_threshold)
        
        print(f"{dataset_name.upper():5} | AUC: {auc:.4f} | F1: {best_f1:.4f} | Threshold: {best_threshold:.3f}")
    
    # Set primary metrics for model comparison
    new_model_score_auc = metric_dict["test_auc"]  # For compatibility with existing logic
    new_model_score_f1 = metric_dict["oot_f1"]    # Primary metric for business decision
    stability_score = metric_dict["oot_auc"]       # Stability check
    
    # Log comparison metrics
    mlflow.log_metric("comparison_metric", new_model_score_auc)  # For compatibility
    mlflow.log_metric("primary_metric_f1", new_model_score_f1)   # Primary business metric
    mlflow.log_metric("stability_metric", stability_score)
    
    # Log the trained model
    mlflow.xgboost.log_model(best_model, artifact_path="model")
    model_uri = f"runs:/{run_id}/model"
    
    # MODEL REGISTRATION LOGIC
    
    registered_model_name = "loan_default_prediction"  # Use your existing name
    
    # Define performance thresholds
    min_f1_threshold = 0.30      # Minimum F1 score for business viability
    min_auc_threshold = 0.65     # Minimum AUC for model stability
    min_improvement = 0.01       # Minimum improvement required over production
    
    print(f"\nModel Performance Summary:")
    print(f"OOT F1 Score: {new_model_score_f1:.4f} (min required: {min_f1_threshold})")
    print(f"OOT AUC Score: {stability_score:.4f} (min required: {min_auc_threshold})")
    print(f"Test AUC Score: {new_model_score_auc:.4f}")
    
    # Check if model meets minimum performance requirements
    meets_requirements = (new_model_score_f1 >= min_f1_threshold and 
                         stability_score >= min_auc_threshold)
    
    if meets_requirements:
        print("Model meets minimum performance requirements")
        
        # Compare against latest registered model
        try:
            client = mlflow.tracking.MlflowClient()
            prod_versions = client.get_latest_versions(registered_model_name, stages=["Production"])
            
            should_register = True
            
            if len(prod_versions) > 0:
                prod_version = prod_versions[0]
                prod_run_id = prod_version.run_id
                
                print(f"Comparing against production model (run: {prod_run_id})")
                
                # Try to fetch F1 metric first (preferred)
                try:
                    prod_f1_history = client.get_metric_history(prod_run_id, "oot_f1")
                    if prod_f1_history:
                        old_model_f1 = prod_f1_history[-1].value
                        improvement = new_model_score_f1 - old_model_f1
                        print(f"F1 Comparison - New: {new_model_score_f1:.4f}, Production: {old_model_f1:.4f}, Improvement: {improvement:.4f}")
                        
                        if improvement < min_improvement:
                            should_register = False
                            print(f"Insufficient F1 improvement ({improvement:.4f} < {min_improvement})")
                        else:
                            print(f"Sufficient F1 improvement ({improvement:.4f} >= {min_improvement})")
                    else:
                        raise Exception("No F1 history found")
                        
                except Exception:
                    # Fall back to AUC comparison if F1 not available
                    try:
                        prod_metric_history = client.get_metric_history(prod_run_id, "comparison_metric")
                        if prod_metric_history:
                            old_model_score = prod_metric_history[-1].value
                            improvement = new_model_score_auc - old_model_score
                            print(f"AUC Comparison - New: {new_model_score_auc:.4f}, Production: {old_model_score:.4f}, Improvement: {improvement:.4f}")
                            
                            if new_model_score_auc <= old_model_score:
                                should_register = False
                                print("New model AUC not better than production")
                            else:
                                print("New model AUC better than production")
                        else:
                            print("No production metrics found, proceeding with registration")
                    except Exception as e:
                        print(f"Could not retrieve production metrics: {e}")
                        print("Proceeding with registration")
            else:
                print("No production model found, proceeding with registration")
            
            # Register model if it should be registered
            if should_register:
                print(f"\nREGISTERING XGBoost model...")
                
                result = mlflow.register_model(
                    model_uri=model_uri,
                    name=registered_model_name,
                    tags={
                        "model_type": "XGBoostClassifier",
                        "oot_f1": str(new_model_score_f1),
                        "oot_auc": str(stability_score),
                        "test_auc": str(new_model_score_auc),
                        "train_date": model_train_date_str,
                        "n_features": str(len(selected_features))
                    }
                )
                
                # Transition to staging first for validation
                client.transition_model_version_stage(
                    name=registered_model_name,
                    version=result.version,
                    stage="Staging",
                    archive_existing_versions=True
                )
                
                print(f"XGBoost model registered and moved to Staging (version {result.version})")
                print(f"   OOT F1: {new_model_score_f1:.4f}")
                print(f"   OOT AUC: {stability_score:.4f}")
                print(f"   Test AUC: {new_model_score_auc:.4f}")
                
                # Log registration decision
                mlflow.log_param("model_registered", True)
                mlflow.log_param("registration_reason", "Model meets requirements and improves over production")
                
            else:
                print(f"\nNOT REGISTERING - Model does not sufficiently improve over production")
                mlflow.log_param("model_registered", False)
                mlflow.log_param("registration_reason", "Insufficient improvement over production model")
                
        except mlflow.exceptions.MlflowException as e:
            print(f"Registry error: {e}. Creating new registered model.")
            result = mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name
            )
            client.transition_model_version_stage(
                name=registered_model_name,
                version=result.version,
                stage="Staging",
                archive_existing_versions=True
            )
            print(f"New registered model created: {registered_model_name} v{result.version}")
            mlflow.log_param("model_registered", True)
            mlflow.log_param("registration_reason", "New model registry created")
    
    else:
        print(f"\nNOT REGISTERING - Model does not meet minimum performance requirements:")
        if new_model_score_f1 < min_f1_threshold:
            print(f"   F1 score {new_model_score_f1:.4f} < {min_f1_threshold}")
        if stability_score < min_auc_threshold:
            print(f"   AUC score {stability_score:.4f} < {min_auc_threshold}")
        
        mlflow.log_param("model_registered", False)
        mlflow.log_param("registration_reason", "Model does not meet minimum performance requirements")

print("\nXGBOOST TRAINING AND REGISTRATION COMPLETED")