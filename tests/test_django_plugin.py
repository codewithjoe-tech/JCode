"""
Tests for the Django edge plugin.

Covers:
- URL routing: path() / re_path() / url()
- Model relations: ForeignKey / OneToOneField / ManyToManyField
- Signal receivers: @receiver(signal, sender=Model)
"""
import textwrap
import tempfile
import os
from pathlib import Path
import pytest

from jcode.domain.models import NodeId, NodeType
from jcode.indexer.plugins.django_plugin import (
    DjangoPlugin,
    EDGE_ROUTE,
    EDGE_REFERENCES,
    EDGE_SIGNAL,
)
from jcode.storage.graph_db import GraphDB
from jcode.storage.object_store import ObjectStore
from jcode.indexer.builder import Indexer
from jcode.indexer.plugins import build_parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _index_source(src: str):
    """Index a snippet and return (graph, store)."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = os.path.join(tmp, "repo")
        os.makedirs(repo)
        # write a fake requirements.txt so django plugin is NOT auto-loaded
        # (we inject it manually via build_parser override)
        with open(os.path.join(repo, "views.py"), "w") as f:
            f.write(textwrap.dedent(src))
        # write django to requirements so auto-detect picks it up
        with open(os.path.join(repo, "requirements.txt"), "w") as f:
            f.write("django>=4.2\n")

        jcode_dir = Path(os.path.join(tmp, ".jcode"))
        store = ObjectStore(jcode_dir)
        graph = GraphDB(jcode_dir)
        parser = build_parser(repo)
        indexer = Indexer(parser, store, graph)
        indexer.index(repo)
        # return detached copies
        all_nodes = list(graph.all_nodes())
        all_edges = list(graph.all_edges())
        return all_nodes, all_edges


# ---------------------------------------------------------------------------
# URL routing
# ---------------------------------------------------------------------------

class TestUrlRouting:
    def test_path_emits_route_edge(self):
        nodes, edges = _index_source("""
            from django.urls import path
            from . import views

            urlpatterns = [
                path('login/', views.login_view, name='login'),
            ]
        """)
        route_edges = [e for e in edges if e.edge_type == EDGE_ROUTE]
        assert route_edges, "Expected at least one ROUTE edge"
        target_ids = {e.target_id for e in route_edges}
        targets = [n for n in nodes if n.id in target_ids]
        names = {n.name for n in targets}
        assert "login_view" in names

    def test_re_path_emits_route_edge(self):
        nodes, edges = _index_source("""
            from django.urls import re_path
            from . import views

            urlpatterns = [
                re_path(r'^api/users/$', views.user_list),
            ]
        """)
        route_edges = [e for e in edges if e.edge_type == EDGE_ROUTE]
        assert route_edges
        target_ids = {e.target_id for e in route_edges}
        targets = [n for n in nodes if n.id in target_ids]
        assert any(n.name == "user_list" for n in targets)

    def test_class_based_view_as_view(self):
        nodes, edges = _index_source("""
            from django.urls import path
            from .views import UserListView

            urlpatterns = [
                path('users/', UserListView.as_view(), name='user-list'),
            ]
        """)
        route_edges = [e for e in edges if e.edge_type == EDGE_ROUTE]
        assert route_edges
        target_ids = {e.target_id for e in route_edges}
        targets = [n for n in nodes if n.id in target_ids]
        names = {n.name for n in targets}
        assert "UserListView" in names

    def test_include_does_not_emit_route(self):
        nodes, edges = _index_source("""
            from django.urls import path, include

            urlpatterns = [
                path('api/', include('myapp.urls')),
            ]
        """)
        route_edges = [e for e in edges if e.edge_type == EDGE_ROUTE]
        assert not route_edges, "include() should not produce a ROUTE edge"


# ---------------------------------------------------------------------------
# Model relations
# ---------------------------------------------------------------------------

class TestModelRelations:
    def test_foreignkey_emits_references_edge(self):
        nodes, edges = _index_source("""
            from django.db import models

            class Post(models.Model):
                author = models.ForeignKey('User', on_delete=models.CASCADE)
        """)
        ref_edges = [e for e in edges if e.edge_type == EDGE_REFERENCES]
        assert ref_edges
        target_ids = {e.target_id for e in ref_edges}
        targets = [n for n in nodes if n.id in target_ids]
        assert any(n.name == "User" for n in targets)

    def test_onetoonefield_emits_references_edge(self):
        nodes, edges = _index_source("""
            from django.db import models

            class Profile(models.Model):
                user = models.OneToOneField(User, on_delete=models.CASCADE)
        """)
        ref_edges = [e for e in edges if e.edge_type == EDGE_REFERENCES]
        assert ref_edges
        target_ids = {e.target_id for e in ref_edges}
        targets = [n for n in nodes if n.id in target_ids]
        assert any(n.name == "User" for n in targets)

    def test_manytomanyfield_emits_references_edge(self):
        nodes, edges = _index_source("""
            from django.db import models

            class Article(models.Model):
                tags = models.ManyToManyField(Tag, blank=True)
        """)
        ref_edges = [e for e in edges if e.edge_type == EDGE_REFERENCES]
        assert ref_edges
        target_ids = {e.target_id for e in ref_edges}
        targets = [n for n in nodes if n.id in target_ids]
        assert any(n.name == "Tag" for n in targets)

    def test_self_reference_ignored(self):
        nodes, edges = _index_source("""
            from django.db import models

            class Category(models.Model):
                parent = models.ForeignKey('self', null=True, on_delete=models.SET_NULL)
        """)
        ref_edges = [e for e in edges if e.edge_type == EDGE_REFERENCES]
        # "self" should be filtered out
        target_ids = {e.target_id for e in ref_edges}
        targets = [n for n in nodes if n.id in target_ids]
        assert not any(n.name == "self" for n in targets)


# ---------------------------------------------------------------------------
# Signal receivers
# ---------------------------------------------------------------------------

class TestSignalReceivers:
    def test_receiver_decorator_emits_signal_edge(self):
        nodes, edges = _index_source("""
            from django.db.models.signals import post_save
            from django.dispatch import receiver
            from .models import User

            @receiver(post_save, sender=User)
            def on_user_saved(sender, instance, created, **kwargs):
                pass
        """)
        sig_edges = [e for e in edges if e.edge_type == EDGE_SIGNAL]
        assert sig_edges
        target_ids = {e.target_id for e in sig_edges}
        targets = [n for n in nodes if n.id in target_ids]
        assert any(n.name == "User" for n in targets)

    def test_receiver_without_sender_no_signal_edge(self):
        nodes, edges = _index_source("""
            from django.db.models.signals import post_save
            from django.dispatch import receiver

            @receiver(post_save)
            def on_any_save(sender, **kwargs):
                pass
        """)
        sig_edges = [e for e in edges if e.edge_type == EDGE_SIGNAL]
        assert not sig_edges, "No sender= means no SIGNAL edge"
