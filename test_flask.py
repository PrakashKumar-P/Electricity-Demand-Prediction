import app

app.app.testing = True
client = app.app.test_client()

def test_login_page():
    response = client.get("/")
    assert response.status_code == 200

def test_dashboard_protection():
    response = client.get("/dashboard")
    assert response.status_code == 302  # redirect if not logged in