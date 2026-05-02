"""Comments domain service."""

from common.utils import get_timestamp, format_timestamp


class CommentService:
    """Handles creation and retrieval of comments."""

    def create_comment(self, text: str, author: str) -> dict:
        """Create a new comment with a timestamp."""
        ts = get_timestamp()
        return {
            "text": text,
            "author": author,
            "created_at": format_timestamp(ts),
        }

    def update_comment(self, comment: dict, new_text: str) -> dict:
        """Update comment text and refresh its timestamp."""
        comment["text"] = new_text
        comment["updated_at"] = format_timestamp(get_timestamp())
        return comment
