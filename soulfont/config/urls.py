"""
URL configuration for the Soul Font project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
import os

from django.contrib import admin #type: ignore
from django.urls import path, include #type: ignore
from django.views.generic import RedirectView #type: ignore
from pybo import views
from django.conf import settings #type: ignore
from django.conf.urls.static import static #type: ignore

urlpatterns = [
    path('', RedirectView.as_view(pattern_name='desktop', permanent=False)),
    path('admin/', admin.site.urls),
    path('pybo/', include('pybo.urls')),
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('create/', views.create_font, name='create_font'),
    path('letter/', views.letter, name='letter'),
    path('learning/', views.learning, name='learning'),
]

# Media (the generated TTFs, profile images) also has nothing in front of it in the
# desktop build. django.views.static.serve is not for a public deployment, but the app
# binds 127.0.0.1 only and serves one local user, and it is hardened against traversal.
if settings.DEBUG or os.environ.get('SOULFONT_APP') == '1':
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)