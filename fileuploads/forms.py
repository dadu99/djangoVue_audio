from django import forms

from fileuploads.models import Document

class UploadForm(forms.ModelForm):
    class Meta:
        model = Document
        fields = ('document', )
