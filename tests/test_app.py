def test_application_imports_and_registers_expected_routes():
    from main import app

    paths = {route.path for route in app.routes}
    assert "/health" in paths
    assert "/api/v1/chat" in paths
    assert "/api/v1/chat/stream" in paths
    assert "/api/v1/advice" in paths
    assert "/api/v1/documents/upload" in paths
    assert "/api/v1/documents/impact" in paths
