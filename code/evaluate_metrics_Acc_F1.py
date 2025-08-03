from sklearn.metrics import accuracy_score, f1_score
import numpy as np


g_truth = ['healthy', 'unhealthy', 'healthy', 'healthy', 'healthy', 'unhealthy', 'unhealthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'unhealthy', 'healthy', 'unhealthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'unhealthy', 'healthy', 'healthy', 'unhealthy', 'healthy', 'unhealthy', 'healthy', 'unhealthy', 'healthy', 'healthy', 'unhealthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'unhealthy', 'healthy', 'unhealthy', 'healthy', 'healthy', 'healthy']

predictions = ['healthy', 'healthy', 'unhealthy', 'healthy', 'healthy', 'healthy', 'unhealthy', 'healthy', 'unhealthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'unhealthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'unhealthy', 'healthy', 'healthy', 'unhealthy', 'healthy', 'unhealthy', 'healthy', 'unhealthy', 'unhealthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'unhealthy', 'healthy', 'unhealthy', 'healthy', 'healthy', 'healthy']



accuracy = accuracy_score(g_truth, predictions) 
f1 = f1_score(g_truth, predictions, pos_label='unhealthy') 

print(f"Ground Truth: {g_truth}")
print(f"Predictions: {predictions}")
print(f"Accuracy: {accuracy:.3f}")
print(f"F1 Score: {f1:.3f}")

