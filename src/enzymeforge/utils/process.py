"""
Process and subprocess utilities for EnzymeForge

Improved from ProtDesign2 with:
- Better error handling
- Timeout support
- Proper logging
- Type hints
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Callable, List, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)


def run_command(
    command: str,
    capture_output: bool = True,
    timeout: Optional[int] = None,
    cwd: Optional[Path] = None
) -> Tuple[int, str, str]:
    """Run shell command with logging and error handling

    Improved from ProtDesign2's run() with:
    - Returns stdout and stderr separately
    - Timeout support
    - Working directory support
    - Better error handling

    Args:
        command: Shell command to execute
        capture_output: If True, capture and return output. If False, stream to console
        timeout: Timeout in seconds (None = no timeout)
        cwd: Working directory (None = current directory)

    Returns:
        Tuple of (return_code, stdout, stderr)

    Raises:
        subprocess.TimeoutExpired: If command times out
        subprocess.CalledProcessError: If command fails and check=True

    Example:
        >>> returncode, stdout, stderr = run_command("ls -la")
        >>> returncode == 0
        True
    """
    logger.info(f"Running command: {command}")

    if capture_output:
        # Capture output for processing
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(cwd) if cwd else None
            )

            if result.returncode != 0:
                logger.error(f"Command failed with return code {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
            else:
                logger.info("Command completed successfully")

            return result.returncode, result.stdout, result.stderr

        except subprocess.TimeoutExpired as e:
            logger.error(f"Command timed out after {timeout} seconds")
            raise

    else:
        # Stream output to console (ProtDesign2 style)
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
            text=True,
            cwd=str(cwd) if cwd else None
        )

        stdout_lines = []
        while True:
            line = process.stdout.readline()
            if line:
                print(f"Output: {line.strip()}")
                stdout_lines.append(line)
            else:
                break

        return_code = process.wait()

        if return_code != 0:
            logger.error(f"Command failed with return code {return_code}")
        else:
            logger.info("Command completed successfully")

        return return_code, "".join(stdout_lines), ""


def run_parallel(
    func: Callable,
    args_list: List[Any],
    max_workers: int = 8,
    task_description: str = "Processing"
) -> List[Any]:
    """Run function in parallel using ProcessPoolExecutor

    Adapted from ProtDesign2's FastRelax parallelization pattern.

    Args:
        func: Function to execute in parallel
        args_list: List of arguments (each item will be unpacked and passed to func)
        max_workers: Maximum number of parallel workers
        task_description: Description for logging

    Returns:
        List of results (in completion order, not submission order)

    Example:
        >>> def square(x):
        ...     return x ** 2
        >>> results = run_parallel(square, [1, 2, 3, 4], max_workers=2)
        >>> sorted(results)
        [1, 4, 9, 16]
    """
    logger.info(f"{task_description}: Starting {len(args_list)} tasks with {max_workers} workers")

    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(func, *args) if isinstance(args, tuple) else executor.submit(func, args)
                   for args in args_list]

        # Collect results as they complete
        completed = 0
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                completed += 1
                logger.info(f"{task_description}: Completed {completed}/{len(args_list)}")
            except Exception as e:
                logger.error(f"{task_description}: Task failed with error: {e}")
                raise

    logger.info(f"{task_description}: All tasks completed")
    return results


def check_executable_exists(executable: str) -> bool:
    """Check if executable is available in PATH

    Args:
        executable: Name of executable (e.g., "colabfold_batch")

    Returns:
        True if executable is found, False otherwise
    """
    try:
        result = subprocess.run(
            ["which", executable],
            capture_output=True,
            text=True
        )
        exists = result.returncode == 0

        if exists:
            logger.info(f"Found executable: {executable} at {result.stdout.strip()}")
        else:
            logger.warning(f"Executable not found: {executable}")

        return exists

    except Exception as e:
        logger.error(f"Error checking for executable {executable}: {e}")
        return False


def check_path_exists(path: Path, path_type: str = "file") -> bool:
    """Check if a path exists and is of the expected type

    Args:
        path: Path to check
        path_type: "file" or "directory"

    Returns:
        True if path exists and is correct type, False otherwise
    """
    if not path.exists():
        logger.error(f"{path_type.capitalize()} not found: {path}")
        return False

    if path_type == "file" and not path.is_file():
        logger.error(f"Path exists but is not a file: {path}")
        return False

    if path_type == "directory" and not path.is_dir():
        logger.error(f"Path exists but is not a directory: {path}")
        return False

    logger.debug(f"Validated {path_type}: {path}")
    return True
