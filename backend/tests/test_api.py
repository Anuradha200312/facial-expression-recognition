def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health_check_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "face_model_loaded" in body
    assert "emotion_model_loaded" in body

def test_predict_endpoint_valid_image(client, sample_valid_image_bytes):
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", sample_valid_image_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert "total_faces" in body
    assert "detections" in body
    assert "execution_time_ms" in body

def test_predict_endpoint_corrupt_image(client, sample_corrupt_bytes):
    response = client.post(
        "/predict",
        files={"file": ("corrupt.jpg", sample_corrupt_bytes, "image/jpeg")}
    )
    assert response.status_code == 400
    assert "Invalid or corrupted image file" in response.json()["detail"]

def test_predict_annotated_endpoint(client, sample_valid_image_bytes):
    response = client.post(
        "/predict-annotated",
        files={"file": ("test.jpg", sample_valid_image_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) > 0

def test_predict_zip_endpoint(client, sample_zip_bytes):
    response = client.post(
        "/predict-zip",
        files={"file": ("batch.zip", sample_zip_bytes, "application/zip")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["total_extracted_images"] == 2
    assert body["total_rejected_files"] == 1
    assert len(body["results"]) == 2
