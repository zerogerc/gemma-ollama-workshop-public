#!/usr/bin/env python3

import asyncio
import os
import sys
from typing import List, Dict, Any
import json
import shutil
import re
from difflib import unified_diff
from datetime import datetime

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.shared.exceptions import McpError
import mcp.types as types
import mcp.server.stdio
from pydantic import BaseModel, Field


class ReadFileArgs(BaseModel):
    path: str


class ReadMultipleFilesArgs(BaseModel):
    paths: List[str]


class WriteFileArgs(BaseModel):
    path: str
    content: str


class EditOperation(BaseModel):
    oldText: str = Field(description="Text to search for - must match exactly")
    newText: str = Field(description="Text to replace with")


class EditFileArgs(BaseModel):
    path: str
    edits: List[EditOperation]
    dryRun: bool = Field(default=False, description="Preview changes using git-style diff format")


class CreateDirectoryArgs(BaseModel):
    path: str


class ListDirectoryArgs(BaseModel):
    path: str


class DirectoryTreeArgs(BaseModel):
    path: str


class MoveFileArgs(BaseModel):
    source: str
    destination: str


class SearchFilesArgs(BaseModel):
    path: str
    pattern: str = Field(description="Regular expression pattern to match file/directory names")


class GetFileInfoArgs(BaseModel):
    path: str


class FileInfo(BaseModel):
    size: int
    created: str
    modified: str
    accessed: str
    isDirectory: bool
    isFile: bool
    permissions: str


def normalize_path(p: str) -> str:
    """Normalize path consistently."""
    return os.path.normpath(p)


def expand_home(filepath: str) -> str:
    """Expand ~ to home directory."""
    if filepath.startswith('~/') or filepath == '~':
        return os.path.join(os.path.expanduser('~'), filepath[1:].lstrip('/'))
    return filepath


async def validate_path(requested_path: str, allowed_directories: List[str]) -> str:
    """Validate that a path is within allowed directories."""
    expanded_path = expand_home(requested_path)
    absolute = os.path.abspath(expanded_path) if not os.path.isabs(expanded_path) else os.path.abspath(expanded_path)
    
    normalized_requested = normalize_path(absolute)
    
    # Check if path is within allowed directories
    is_allowed = any(normalized_requested.startswith(dir_path) for dir_path in allowed_directories)
    if not is_allowed:
        raise McpError(
            types.ErrorData(
                code=types.INVALID_PARAMS,
                message=f"Access denied - path outside allowed directories: {absolute} not in {', '.join(allowed_directories)}"
            )
        )
    
    # Handle symlinks by checking their real path
    try:
        real_path = os.path.realpath(absolute)
        normalized_real = normalize_path(real_path)
        is_real_path_allowed = any(normalized_real.startswith(dir_path) for dir_path in allowed_directories)
        if not is_real_path_allowed:
            raise McpError(
                types.ErrorData(
                    code=types.INVALID_PARAMS,
                    message="Access denied - symlink target outside allowed directories"
                )
            )
        return real_path
    except Exception:
        # For new files that don't exist yet, verify parent directory
        parent_dir = os.path.dirname(absolute)
        try:
            real_parent_path = os.path.realpath(parent_dir)
            normalized_parent = normalize_path(real_parent_path)
            is_parent_allowed = any(normalized_parent.startswith(dir_path) for dir_path in allowed_directories)
            if not is_parent_allowed:
                raise McpError(
                    types.ErrorData(
                        code=types.INVALID_PARAMS,
                        message="Access denied - parent directory outside allowed directories"
                    )
                )
            return absolute
        except Exception:
            raise McpError(
                types.ErrorData(
                    code=types.INVALID_PARAMS,
                    message=f"Parent directory does not exist: {parent_dir}"
                )
            )


async def get_file_stats(file_path: str) -> FileInfo:
    """Get detailed file statistics."""
    try:
        stats = os.stat(file_path)
        return FileInfo(
            size=stats.st_size,
            created=datetime.fromtimestamp(stats.st_ctime).isoformat(),
            modified=datetime.fromtimestamp(stats.st_mtime).isoformat(),
            accessed=datetime.fromtimestamp(stats.st_atime).isoformat(),
            isDirectory=os.path.isdir(file_path),
            isFile=os.path.isfile(file_path),
            permissions=oct(stats.st_mode)[-3:]
        )
    except Exception as e:
        raise McpError(
            types.ErrorData(
                code=types.INTERNAL_ERROR,
                message=f"Failed to get file stats: {str(e)}"
            )
        )


async def search_files(root_path: str, pattern: str, allowed_directories: List[str]) -> List[str]:
    """Recursively search for files matching a regex pattern."""
    results = []
    
    # Compile regex pattern for efficiency
    try:
        compiled_pattern = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        raise McpError(
            types.ErrorData(
                code=types.INVALID_PARAMS,
                message=f"Invalid regex pattern '{pattern}': {str(e)}"
            )
        )
    
    async def search_recursive(current_path: str):
        try:
            for entry in os.listdir(current_path):
                full_path = os.path.join(current_path, entry)
                
                try:
                    # Validate each path before processing
                    await validate_path(full_path, allowed_directories)
                    
                    # Check if entry name matches the search pattern
                    if compiled_pattern.search(entry):
                        results.append(full_path)
                    
                    # Recursively search directories
                    if os.path.isdir(full_path):
                        await search_recursive(full_path)
                        
                except Exception:
                    # Skip invalid paths during search
                    continue
                    
        except Exception:
            # Skip directories we can't read
            pass
    
    await search_recursive(root_path)
    return results


def normalize_line_endings(text: str) -> str:
    """Normalize line endings to LF."""
    return text.replace('\r\n', '\n')


def create_unified_diff(original_content: str, new_content: str, filepath: str = 'file') -> str:
    """Create a unified diff between two content strings."""
    normalized_original = normalize_line_endings(original_content)
    normalized_new = normalize_line_endings(new_content)
    
    diff_lines = list(unified_diff(
        normalized_original.splitlines(keepends=True),
        normalized_new.splitlines(keepends=True),
        fromfile=f"{filepath} (original)",
        tofile=f"{filepath} (modified)",
        lineterm=""
    ))
    
    return ''.join(diff_lines)


async def apply_file_edits(file_path: str, edits: List[EditOperation], dry_run: bool = False) -> str:
    """Apply line-based edits to a file and return diff."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = normalize_line_endings(f.read())
    except Exception as e:
        raise McpError(
            types.ErrorData(
                code=types.INTERNAL_ERROR,
                message=f"Failed to read file: {str(e)}"
            )
        )
    
    # Apply edits sequentially
    modified_content = content
    for edit in edits:
        normalized_old = normalize_line_endings(edit.oldText)
        normalized_new = normalize_line_endings(edit.newText)
        
        # If exact match exists, use it
        if normalized_old in modified_content:
            modified_content = modified_content.replace(normalized_old, normalized_new, 1)
            continue
        
        # Otherwise, try line-by-line matching with flexibility for whitespace
        old_lines = normalized_old.split('\n')
        content_lines = modified_content.split('\n')
        match_found = False
        
        for i in range(len(content_lines) - len(old_lines) + 1):
            potential_match = content_lines[i:i + len(old_lines)]
            
            # Compare lines with normalized whitespace
            is_match = all(
                old_line.strip() == content_line.strip()
                for old_line, content_line in zip(old_lines, potential_match)
            )
            
            if is_match:
                # Preserve original indentation of first line
                original_indent = ''
                if content_lines[i]:
                    original_indent = content_lines[i][:len(content_lines[i]) - len(content_lines[i].lstrip())]
                
                new_lines = normalized_new.split('\n')
                formatted_new_lines = []
                for j, line in enumerate(new_lines):
                    if j == 0:
                        formatted_new_lines.append(original_indent + line.lstrip())
                    else:
                        # For subsequent lines, try to preserve relative indentation
                        if j < len(old_lines):
                            old_indent = old_lines[j][:len(old_lines[j]) - len(old_lines[j].lstrip())] if old_lines[j] else ''
                            new_indent = line[:len(line) - len(line.lstrip())] if line else ''
                            if old_indent and new_indent:
                                relative_indent_diff = len(new_indent) - len(old_indent)
                                final_indent = original_indent + ' ' * max(0, relative_indent_diff)
                                formatted_new_lines.append(final_indent + line.lstrip())
                            else:
                                formatted_new_lines.append(line)
                        else:
                            formatted_new_lines.append(line)
                
                content_lines[i:i + len(old_lines)] = formatted_new_lines
                modified_content = '\n'.join(content_lines)
                match_found = True
                break
        
        if not match_found:
            raise McpError(
                types.ErrorData(
                    code=types.INVALID_PARAMS,
                    message=f"Could not find exact match for edit:\n{edit.oldText}"
                )
            )
    
    # Create unified diff
    diff = create_unified_diff(content, modified_content, file_path)
    
    # Format diff with appropriate number of backticks
    num_backticks = 3
    while '`' * num_backticks in diff:
        num_backticks += 1
    formatted_diff = f"{'`' * num_backticks}diff\n{diff}{'`' * num_backticks}\n\n"
    
    if not dry_run:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
        except Exception as e:
            raise McpError(
                types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Failed to write file: {str(e)}"
                )
            )
    
    return formatted_diff


async def serve(allowed_directories: List[str]) -> None:
    """Run the secure filesystem MCP server."""
    # Validate that all directories exist and are accessible
    normalized_dirs = []
    for dir_path in allowed_directories:
        expanded = expand_home(dir_path)
        if not os.path.exists(expanded):
            raise ValueError(f"Directory does not exist: {dir_path}")
        if not os.path.isdir(expanded):
            raise ValueError(f"Path is not a directory: {dir_path}")
        normalized_dirs.append(normalize_path(os.path.abspath(expanded)))
    
    server = Server("filesystem-server")
    
    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        """List available tools."""
        return [
            types.Tool(
                name="read_file",
                description=(
                    "Read the complete contents of a file from the file system. "
                    "Handles various text encodings and provides detailed error messages "
                    "if the file cannot be read. Use this tool when you need to examine "
                    "the contents of a single file. Only works within allowed directories."
                ),
                inputSchema=ReadFileArgs.model_json_schema(),
            ),
            types.Tool(
                name="read_multiple_files",
                description=(
                    "Read the contents of multiple files simultaneously. This is more "
                    "efficient than reading files one by one when you need to analyze "
                    "or compare multiple files. Each file's content is returned with its "
                    "path as a reference. Failed reads for individual files won't stop "
                    "the entire operation. Only works within allowed directories."
                ),
                inputSchema=ReadMultipleFilesArgs.model_json_schema(),
            ),
            types.Tool(
                name="write_file",
                description=(
                    "Create a new file or completely overwrite an existing file with new content. "
                    "Use with caution as it will overwrite existing files without warning. "
                    "Handles text content with proper encoding. Only works within allowed directories."
                ),
                inputSchema=WriteFileArgs.model_json_schema(),
            ),
            types.Tool(
                name="edit_file",
                description=(
                    "Make line-based edits to a text file. Each edit replaces exact line sequences "
                    "with new content. Returns a git-style diff showing the changes made. "
                    "Only works within allowed directories."
                ),
                inputSchema=EditFileArgs.model_json_schema(),
            ),
            types.Tool(
                name="create_directory",
                description=(
                    "Create a new directory or ensure a directory exists. Can create multiple "
                    "nested directories in one operation. If the directory already exists, "
                    "this operation will succeed silently. Perfect for setting up directory "
                    "structures for projects or ensuring required paths exist. Only works within allowed directories."
                ),
                inputSchema=CreateDirectoryArgs.model_json_schema(),
            ),
            types.Tool(
                name="list_directory",
                description=(
                    "Get a detailed listing of all files and directories in a specified path. "
                    "Results clearly distinguish between files and directories with [FILE] and [DIR] "
                    "prefixes. This tool is essential for understanding directory structure and "
                    "finding specific files within a directory. Only works within allowed directories."
                ),
                inputSchema=ListDirectoryArgs.model_json_schema(),
            ),
            types.Tool(
                name="directory_tree",
                description=(
                    "Get a recursive tree view of files and directories as a JSON structure. "
                    "Each entry includes 'name', 'type' (file/directory), and 'children' for directories. "
                    "Files have no children array, while directories always have a children array (which may be empty). "
                    "The output is formatted with 2-space indentation for readability. Only works within allowed directories."
                ),
                inputSchema=DirectoryTreeArgs.model_json_schema(),
            ),
            types.Tool(
                name="move_file",
                description=(
                    "Move or rename files and directories. Can move files between directories "
                    "and rename them in a single operation. If the destination exists, the "
                    "operation will fail. Works across different directories and can be used "
                    "for simple renaming within the same directory. Both source and destination must be within allowed directories."
                ),
                inputSchema=MoveFileArgs.model_json_schema(),
            ),
            types.Tool(
                name="search_files",
                description=(
                    "Recursively search for files and directories matching a regular expression pattern. "
                    "Searches through all subdirectories from the starting path. The search "
                    "is case-insensitive and uses regex for flexible pattern matching. Returns full paths to all "
                    "matching items. Great for finding files when you don't know their exact location. "
                    "Only searches within allowed directories."
                ),
                inputSchema=SearchFilesArgs.model_json_schema(),
            ),
            types.Tool(
                name="get_file_info",
                description=(
                    "Retrieve detailed metadata about a file or directory. Returns comprehensive "
                    "information including size, creation time, last modified time, permissions, "
                    "and type. This tool is perfect for understanding file characteristics "
                    "without reading the actual content. Only works within allowed directories."
                ),
                inputSchema=GetFileInfoArgs.model_json_schema(),
            ),
            types.Tool(
                name="list_allowed_directories",
                description=(
                    "Returns the list of directories that this server is allowed to access. "
                    "Use this to understand which directories are available before trying to access files."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
        ]
    
    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: dict | None
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        """Handle tool execution."""
        if not arguments:
            arguments = {}
        
        try:
            if name == "read_file":
                args = ReadFileArgs(**arguments)
                valid_path = await validate_path(args.path, normalized_dirs)
                
                try:
                    with open(valid_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    return [types.TextContent(type="text", text=content)]
                except Exception as e:
                    raise McpError(
                        types.ErrorData(
                            code=types.INTERNAL_ERROR,
                            message=f"Failed to read file: {str(e)}"
                        )
                    )
            
            elif name == "read_multiple_files":
                args = ReadMultipleFilesArgs(**arguments)
                results = []
                
                for file_path in args.paths:
                    try:
                        valid_path = await validate_path(file_path, normalized_dirs)
                        with open(valid_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        results.append(f"{file_path}:\n{content}\n")
                    except Exception as e:
                        results.append(f"{file_path}: Error - {str(e)}")
                
                return [types.TextContent(type="text", text="\n---\n".join(results))]
            
            elif name == "write_file":
                args = WriteFileArgs(**arguments)
                valid_path = await validate_path(args.path, normalized_dirs)
                
                try:
                    with open(valid_path, 'w', encoding='utf-8') as f:
                        f.write(args.content)
                    return [types.TextContent(type="text", text=f"Successfully wrote to {args.path}")]
                except Exception as e:
                    raise McpError(
                        types.ErrorData(
                            code=types.INTERNAL_ERROR,
                            message=f"Failed to write file: {str(e)}"
                        )
                    )
            
            elif name == "edit_file":
                args = EditFileArgs(**arguments)
                valid_path = await validate_path(args.path, normalized_dirs)
                result = await apply_file_edits(valid_path, args.edits, args.dryRun)
                return [types.TextContent(type="text", text=result)]
            
            elif name == "create_directory":
                args = CreateDirectoryArgs(**arguments)
                valid_path = await validate_path(args.path, normalized_dirs)
                
                try:
                    os.makedirs(valid_path, exist_ok=True)
                    return [types.TextContent(type="text", text=f"Successfully created directory {args.path}")]
                except Exception as e:
                    raise McpError(
                        types.ErrorData(
                            code=types.INTERNAL_ERROR,
                            message=f"Failed to create directory: {str(e)}"
                        )
                    )
            
            elif name == "list_directory":
                args = ListDirectoryArgs(**arguments)
                valid_path = await validate_path(args.path, normalized_dirs)
                
                try:
                    entries = []
                    for entry in os.listdir(valid_path):
                        full_path = os.path.join(valid_path, entry)
                        if os.path.isdir(full_path):
                            entries.append(f"[DIR] {entry}")
                        else:
                            entries.append(f"[FILE] {entry}")
                    
                    return [types.TextContent(type="text", text="\n".join(entries))]
                except Exception as e:
                    raise McpError(
                        types.ErrorData(
                            code=types.INTERNAL_ERROR,
                            message=f"Failed to list directory: {str(e)}"
                        )
                    )
            
            elif name == "directory_tree":
                args = DirectoryTreeArgs(**arguments)
                valid_path = await validate_path(args.path, normalized_dirs)
                
                async def build_tree(current_path: str) -> List[Dict[str, Any]]:
                    result = []
                    try:
                        for entry in os.listdir(current_path):
                            entry_path = os.path.join(current_path, entry)
                            try:
                                await validate_path(entry_path, normalized_dirs)
                                entry_data = {
                                    "name": entry,
                                    "type": "directory" if os.path.isdir(entry_path) else "file"
                                }
                                
                                if os.path.isdir(entry_path):
                                    entry_data["children"] = await build_tree(entry_path)
                                
                                result.append(entry_data)
                            except Exception:
                                continue
                    except Exception:
                        pass
                    return result
                
                tree_data = await build_tree(valid_path)
                return [types.TextContent(type="text", text=json.dumps(tree_data, indent=2))]
            
            elif name == "move_file":
                args = MoveFileArgs(**arguments)
                valid_source = await validate_path(args.source, normalized_dirs)
                valid_dest = await validate_path(args.destination, normalized_dirs)
                
                try:
                    shutil.move(valid_source, valid_dest)
                    return [types.TextContent(type="text", text=f"Successfully moved {args.source} to {args.destination}")]
                except Exception as e:
                    raise McpError(
                        types.ErrorData(
                            code=types.INTERNAL_ERROR,
                            message=f"Failed to move file: {str(e)}"
                        )
                    )
            
            elif name == "search_files":
                args = SearchFilesArgs(**arguments)
                valid_path = await validate_path(args.path, normalized_dirs)
                results = await search_files(valid_path, args.pattern, normalized_dirs)
                
                if results:
                    return [types.TextContent(type="text", text="\n".join(results))]
                else:
                    return [types.TextContent(type="text", text="No matches found")]
            
            elif name == "get_file_info":
                args = GetFileInfoArgs(**arguments)
                valid_path = await validate_path(args.path, normalized_dirs)
                info = await get_file_stats(valid_path)
                
                info_text = "\n".join([
                    f"size: {info.size}",
                    f"created: {info.created}",
                    f"modified: {info.modified}",
                    f"accessed: {info.accessed}",
                    f"isDirectory: {info.isDirectory}",
                    f"isFile: {info.isFile}",
                    f"permissions: {info.permissions}"
                ])
                
                return [types.TextContent(type="text", text=info_text)]
            
            elif name == "list_allowed_directories":
                dir_list = "Allowed directories:\n" + "\n".join(normalized_dirs)
                return [types.TextContent(type="text", text=dir_list)]
            
            else:
                raise McpError(
                    types.ErrorData(
                        code=types.INVALID_PARAMS,
                        message=f"Unknown tool: {name}"
                    )
                )
        
        except McpError:
            raise
        except Exception as e:
            raise McpError(
                types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Tool execution failed: {str(e)}"
                )
            )
    
    # Run the server
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="filesystem-server",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python server.py <allowed-directory> [additional-directories...]", file=sys.stderr)
        sys.exit(1)
    
    allowed_dirs = sys.argv[1:]
    print(f"Secure MCP Filesystem Server running on stdio", file=sys.stderr)
    print(f"Allowed directories: {allowed_dirs}", file=sys.stderr)
    
    try:
        asyncio.run(serve(allowed_dirs))
    except Exception as error:
        print(f"Fatal error running server: {error}", file=sys.stderr)
        sys.exit(1)