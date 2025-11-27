"""Agent calendar package

Initial package layout to group calendar/availability tools by domain.

This module re-exports time off tools from the existing implementation to
avoid any functional change while establishing the package pattern. Future
PRs can migrate the underlying implementations into submodules here.
"""

from .timeoff import (
    ToolAddTimeOffIn,
    ToolAddTimeOffOut,
    ToolUpdateTimeOffIn,
    ToolUpdateTimeOffOut,
    ToolListTimeOffIn,
    ToolListTimeOffOut,
    ToolNextTimeOffOut,
    ToolDeleteTimeOffIn,
    ToolDeleteTimeOffOut,
    add_time_off_tool,
    update_time_off_tool,
    list_time_off_tool,
    next_time_off_tool,
    delete_time_off_tool,
)

__all__ = [
    "ToolAddTimeOffIn",
    "ToolAddTimeOffOut",
    "ToolUpdateTimeOffIn",
    "ToolUpdateTimeOffOut",
    "ToolListTimeOffIn",
    "ToolListTimeOffOut",
    "ToolNextTimeOffOut",
    "ToolDeleteTimeOffIn",
    "ToolDeleteTimeOffOut",
    "add_time_off_tool",
    "update_time_off_tool",
    "list_time_off_tool",
    "next_time_off_tool",
    "delete_time_off_tool",
]
