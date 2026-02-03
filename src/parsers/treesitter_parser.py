from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_repository(repo_path: str) -> list[dict]:
    """Parse code and return a list of structured units.

    TODO: integrate tree-sitter to extract functions, classes, imports, and calls.
    """
    logger.info("Parsing repository at %s", repo_path)
    return []
