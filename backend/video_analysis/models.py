from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()

class VideoAnalysis(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='video_analyses')
    video_file = models.FileField(upload_to='videos/')
    filename = models.CharField(max_length=255)
    file_size = models.BigIntegerField()  # in bytes
    
    # Analysis results
    prediction_score = models.FloatField()
    is_deepfake = models.BooleanField()
    confidence_threshold = models.FloatField(default=0.5)
    
    # Metadata
    analysis_duration = models.FloatField(null=True, blank=True)  # in seconds
    frames_analyzed = models.IntegerField(null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        
    def __str__(self):
        return f"{self.filename} - {'Deepfake' if self.is_deepfake else 'Authentic'}"
    
    @property
    def confidence_percentage(self):
        """Return confidence as percentage"""
        if self.is_deepfake:
            return self.prediction_score * 100
        else:
            return (1 - self.prediction_score) * 100