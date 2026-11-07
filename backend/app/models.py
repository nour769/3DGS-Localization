from datetime import datetime, timezone
from uuid import uuid4

from .extensions import db


class Localization(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid4()))
    image_name = db.Column(db.String(255), nullable=False)
    image_path = db.Column(db.Text, nullable=False)
    scene_id = db.Column(db.String(255), nullable=False, index=True)
    status = db.Column(db.String(24), nullable=False, default="processing", index=True)
    pose_json = db.Column(db.JSON, nullable=True)
    metrics_json = db.Column(db.JSON, nullable=True)
    error_message = db.Column(db.Text, nullable=True)
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )

    def to_dict(self):
        return {
            "id": self.id,
            "image_name": self.image_name,
            "scene_id": self.scene_id,
            "status": self.status,
            "pose": self.pose_json,
            "metrics": self.metrics_json or {},
            "error": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
