from django.db import models

# Create your models here.
class Algorithm(models.Model):
    name = models.CharField(max_length=60)

    def __str__(self):
        return self.name


class AlgorithmPostEvent(models.Model):
    algorithm_code = models.CharField(max_length=60)
    aiserving_version = models.CharField(max_length=60)
    post_num = models.PositiveIntegerField()

    def __str__(self):
        return self.algorithm_code
