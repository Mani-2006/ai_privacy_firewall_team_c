def flatten_summary(summary):
    if isinstance(summary, dict):
        return summary.get('value') or summary.get('description') or str(summary)
    return summary 