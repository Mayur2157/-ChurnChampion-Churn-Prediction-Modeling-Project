# utils.py

def report_performance(y_true, y_pred):
    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
