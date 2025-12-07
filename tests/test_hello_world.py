def test_mlflow_import():
    try:
        import mlflow
    except ImportError:
        assert False, "mlflow no está instalado"