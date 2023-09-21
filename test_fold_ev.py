from fold_evaluator import FoldEvaluator

if __name__ == "__main__":
    c = FoldEvaluator(prefix="all", folds=10, algorithms=["mlr","svr","ann","ann_es"])
    c.process()