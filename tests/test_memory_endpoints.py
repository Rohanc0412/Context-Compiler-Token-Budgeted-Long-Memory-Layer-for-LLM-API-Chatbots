def test_memory_export_and_delete(client):
    user_id = "u3"
    session_id = "s3"
    resp = client.post(
        "/events",
        json={
            "user_id": user_id,
            "session_id": session_id,
            "role": "user",
            "text": "I must avoid peanuts.",
        },
    )
    assert resp.status_code == 200
    export_resp = client.get(f"/memory/export?user_id={user_id}")
    data = export_resp.json()
    assert len(data["memories"]) >= 1
    del_resp = client.post("/memory/delete", json={"user_id": user_id, "session_id": session_id})
    assert del_resp.status_code == 200
    export_after = client.get(f"/memory/export?user_id={user_id}")
    assert export_after.json()["memories"] == []
