from django.db import models

# Create your models here.

class Song(models.Model):
    lyrics = models.TextField(blank=False, null=False)
