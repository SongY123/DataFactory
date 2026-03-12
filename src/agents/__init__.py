from __future__ import annotations

__all__ = [
    "create_table_finder_worker",
    "create_data_analyst_worker",
    "create_business_consultant_worker",
    "create_visualization_worker",
    "create_report_writer_worker",
    "create_orchestrator",
]


def create_table_finder_worker(*args, **kwargs):
    from .table_finder_worker import create_table_finder_worker as impl

    return impl(*args, **kwargs)


def create_data_analyst_worker(*args, **kwargs):
    from .data_analyst_worker import create_data_analyst_worker as impl

    return impl(*args, **kwargs)


def create_business_consultant_worker(*args, **kwargs):
    from .business_consultant_worker import create_business_consultant_worker as impl

    return impl(*args, **kwargs)


def create_visualization_worker(*args, **kwargs):
    from .visualization_worker import create_visualization_worker as impl

    return impl(*args, **kwargs)


def create_report_writer_worker(*args, **kwargs):
    from .report_writer_worker import create_report_writer_worker as impl

    return impl(*args, **kwargs)


def create_orchestrator(*args, **kwargs):
    from .orchestrator import create_orchestrator as impl

    return impl(*args, **kwargs)
