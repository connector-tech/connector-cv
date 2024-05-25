from tortoise import models, fields


class UserSessionPhoto(models.Model):
    id = fields.UUIDField(pk=True)
    session_id = fields.UUIDField()
    user_id = fields.UUIDField()
    photo_id = fields.UUIDField()
    bestframe_score = fields.FloatField()
    liveness_score = fields.FloatField()
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = 'user_session_photo'
