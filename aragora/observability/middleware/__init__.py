"""Observability middleware package.

Houses request-scoped observability middleware (distributed tracing, request
correlation, the OpenTelemetry bridge, and request logging) at the
infrastructure layer so foundation/infrastructure modules can use them without
reaching up into the interface (server) layer.

The deprecated re-export shims under the server middleware package forward here.
"""
