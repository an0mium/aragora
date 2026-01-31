"""
Tests for Gastown agent roles adapter module.

Tests re-exports from nomic agent roles layer.
"""

from __future__ import annotations

import pytest


class TestAgentRoleExports:
    """Tests for agent_roles module re-exports."""

    def test_agent_hierarchy_exported(self):
        """AgentHierarchy class is exported."""
        from aragora.extensions.gastown.agent_roles import AgentHierarchy

        assert AgentHierarchy is not None

    def test_agent_role_exported(self):
        """AgentRole class is exported."""
        from aragora.extensions.gastown.agent_roles import AgentRole

        assert AgentRole is not None

    def test_role_assignment_exported(self):
        """RoleAssignment class is exported."""
        from aragora.extensions.gastown.agent_roles import RoleAssignment

        assert RoleAssignment is not None

    def test_role_based_router_exported(self):
        """RoleBasedRouter class is exported."""
        from aragora.extensions.gastown.agent_roles import RoleBasedRouter

        assert RoleBasedRouter is not None

    def test_role_capability_exported(self):
        """RoleCapability class is exported."""
        from aragora.extensions.gastown.agent_roles import RoleCapability

        assert RoleCapability is not None

    def test_all_exports_in_module_all(self):
        """All documented exports are in __all__."""
        from aragora.extensions.gastown import agent_roles

        expected = {
            "AgentHierarchy",
            "AgentRole",
            "RoleAssignment",
            "RoleBasedRouter",
            "RoleCapability",
        }
        assert set(agent_roles.__all__) == expected


class TestAgentRoleIntegration:
    """Tests for agent role integration between layers."""

    def test_agent_hierarchy_is_same_class(self):
        """Gastown AgentHierarchy is same class as nomic AgentHierarchy."""
        from aragora.extensions.gastown.agent_roles import AgentHierarchy as GastownHierarchy
        from aragora.nomic.agent_roles import AgentHierarchy as NomicHierarchy

        assert GastownHierarchy is NomicHierarchy

    def test_agent_role_is_same_class(self):
        """Gastown AgentRole is same class as nomic AgentRole."""
        from aragora.extensions.gastown.agent_roles import AgentRole as GastownRole
        from aragora.nomic.agent_roles import AgentRole as NomicRole

        assert GastownRole is NomicRole

    def test_role_assignment_is_same_class(self):
        """Gastown RoleAssignment is same class as nomic RoleAssignment."""
        from aragora.extensions.gastown.agent_roles import RoleAssignment as GastownAssignment
        from aragora.nomic.agent_roles import RoleAssignment as NomicAssignment

        assert GastownAssignment is NomicAssignment

    def test_role_based_router_is_same_class(self):
        """Gastown RoleBasedRouter is same class as nomic RoleBasedRouter."""
        from aragora.extensions.gastown.agent_roles import RoleBasedRouter as GastownRouter
        from aragora.nomic.agent_roles import RoleBasedRouter as NomicRouter

        assert GastownRouter is NomicRouter

    def test_role_capability_is_same_class(self):
        """Gastown RoleCapability is same class as nomic RoleCapability."""
        from aragora.extensions.gastown.agent_roles import RoleCapability as GastownCap
        from aragora.nomic.agent_roles import RoleCapability as NomicCap

        assert GastownCap is NomicCap
