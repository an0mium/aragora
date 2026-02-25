"""
SCIM Filter Parser.

Implements RFC 7644 Section 3.4.2.2 for parsing SCIM filter expressions.

Supports:
- Attribute operators: eq, ne, co, sw, ew, pr, gt, ge, lt, le
- Logical operators: and, or, not
- Grouping: parentheses
- Value paths: attribute[filter]

Example filters:
- userName eq "john@example.com"
- active eq true
- emails[type eq "work"].value co "@example.com"
- name.familyName sw "J"
- created gt "2023-01-01T00:00:00Z"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class SCIMOperator(str, Enum):
    """SCIM filter operators."""

    EQ = "eq"  # Equal
    NE = "ne"  # Not equal
    CO = "co"  # Contains
    SW = "sw"  # Starts with
    EW = "ew"  # Ends with
    PR = "pr"  # Present (has value)
    GT = "gt"  # Greater than
    GE = "ge"  # Greater than or equal
    LT = "lt"  # Less than
    LE = "le"  # Less than or equal


class SCIMLogicalOperator(str, Enum):
    """SCIM logical operators."""

    AND = "and"
    OR = "or"
    NOT = "not"


@dataclass
class SCIMFilter:
    """A parsed SCIM filter expression."""

    attribute: str
    operator: SCIMOperator
    value: Any
    sub_attribute: str | None = None

    def matches(self, resource: dict[str, Any]) -> bool:
        """
        Check if a resource matches this filter.

        Args:
            resource: The SCIM resource as a dictionary

        Returns:
            True if the resource matches the filter
        """
        # Get the attribute value from resource
        attr_value = self._get_attribute_value(resource, self.attribute, self.sub_attribute)

        # Handle presence operator
        if self.operator == SCIMOperator.PR:
            return attr_value is not None and attr_value != ""

        # Handle null values
        if attr_value is None:
            return self.operator == SCIMOperator.NE and self.value is not None

        # String comparisons (case-insensitive for most attributes)
        if isinstance(attr_value, str) and isinstance(self.value, str):
            attr_lower = attr_value.lower()
            val_lower = self.value.lower()

            if self.operator == SCIMOperator.EQ:
                return attr_lower == val_lower
            elif self.operator == SCIMOperator.NE:
                return attr_lower != val_lower
            elif self.operator == SCIMOperator.CO:
                return val_lower in attr_lower
            elif self.operator == SCIMOperator.SW:
                return attr_lower.startswith(val_lower)
            elif self.operator == SCIMOperator.EW:
                return attr_lower.endswith(val_lower)
            elif self.operator == SCIMOperator.GT:
                return attr_value > self.value
            elif self.operator == SCIMOperator.GE:
                return attr_value >= self.value
            elif self.operator == SCIMOperator.LT:
                return attr_value < self.value
            elif self.operator == SCIMOperator.LE:
                return attr_value <= self.value

        # Boolean comparisons
        if isinstance(attr_value, bool):
            filter_bool = (
                self.value if isinstance(self.value, bool) else str(self.value).lower() == "true"
            )
            if self.operator == SCIMOperator.EQ:
                return attr_value == filter_bool
            elif self.operator == SCIMOperator.NE:
                return attr_value != filter_bool

        # Numeric comparisons
        if isinstance(attr_value, (int, float)) and isinstance(self.value, (int, float, str)):
            try:
                filter_val = float(self.value) if isinstance(self.value, str) else self.value
                if self.operator == SCIMOperator.EQ:
                    return attr_value == filter_val
                elif self.operator == SCIMOperator.NE:
                    return attr_value != filter_val
                elif self.operator == SCIMOperator.GT:
                    return attr_value > filter_val
                elif self.operator == SCIMOperator.GE:
                    return attr_value >= filter_val
                elif self.operator == SCIMOperator.LT:
                    return attr_value < filter_val
                elif self.operator == SCIMOperator.LE:
                    return attr_value <= filter_val
            except ValueError:
                return False

        # Default equality check
        if self.operator == SCIMOperator.EQ:
            return attr_value == self.value
        elif self.operator == SCIMOperator.NE:
            return attr_value != self.value

        return False

    def _get_attribute_value(
        self,
        resource: dict[str, Any],
        attribute: str,
        sub_attribute: str | None = None,
    ) -> Any:
        """Get an attribute value from a resource, handling nested paths."""
        # Handle dotted paths (e.g., name.familyName)
        parts = attribute.split(".")

        value = resource
        for part in parts:
            if isinstance(value, dict):
                # Handle camelCase conversion
                value = value.get(part) or value.get(self._to_camel_case(part))
            else:
                return None

        # Handle multi-valued attributes (e.g., emails)
        if isinstance(value, list) and sub_attribute:
            for item in value:
                if isinstance(item, dict):
                    sub_val = item.get(sub_attribute) or item.get(
                        self._to_camel_case(sub_attribute)
                    )
                    if sub_val is not None:
                        return sub_val
            return None

        return value

    def _to_camel_case(self, snake_str: str) -> str:
        """Convert snake_case to camelCase."""
        components = snake_str.split("_")
        return components[0] + "".join(x.title() for x in components[1:])


@dataclass
class SCIMCompoundFilter:
    """A compound filter with logical operators."""

    operator: SCIMLogicalOperator
    filters: list[SCIMFilter | SCIMCompoundFilter]

    def matches(self, resource: dict[str, Any]) -> bool:
        """Check if a resource matches this compound filter."""
        if self.operator == SCIMLogicalOperator.AND:
            return all(f.matches(resource) for f in self.filters)
        elif self.operator == SCIMLogicalOperator.OR:
            return any(f.matches(resource) for f in self.filters)
        elif self.operator == SCIMLogicalOperator.NOT:
            return not self.filters[0].matches(resource) if self.filters else True
        return False


class SCIMFilterParser:
    """
    Parser for SCIM filter expressions.

    Usage:
        parser = SCIMFilterParser()
        filter = parser.parse('userName eq "john"')
        if filter.matches(user_dict):
            print("Match!")
    """

    # Token patterns
    ATTRIBUTE = r"[a-zA-Z][a-zA-Z0-9_]*(?:\.[a-zA-Z][a-zA-Z0-9_]*)*"
    OPERATOR = r"(?:eq|ne|co|sw|ew|pr|gt|ge|lt|le)"
    STRING_VALUE = r'"(?:[^"\\]|\\.)*"'
    BOOL_VALUE = r"(?:true|false)"
    NUMBER_VALUE = r"-?\d+(?:\.\d+)?"
    LOGICAL = r"(?:and|or|not)"

    def __init__(self) -> None:
        """Initialize the filter parser."""
        self._pos = 0
        self._text = ""
        self._tokens: list[tuple[str, Any]] = []

    def parse(self, filter_text: str) -> SCIMFilter | SCIMCompoundFilter | None:
        """
        Parse a SCIM filter expression.

        Args:
            filter_text: The filter expression string

        Returns:
            A SCIMFilter or SCIMCompoundFilter object, or None if empty

        Raises:
            ValueError: If the filter syntax is invalid
        """
        if not filter_text or not filter_text.strip():
            return None

        self._text = filter_text.strip()
        self._pos = 0
        self._tokenize()

        if not self._tokens:
            return None

        return self._parse_expression()

    def _tokenize(self) -> None:
        """Tokenize the filter expression."""
        self._tokens = []
        pos = 0
        text = self._text

        while pos < len(text):
            # Skip whitespace
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if pos >= len(text):
                break

            # Check for parentheses
            if text[pos] == "(":
                self._tokens.append(("LPAREN", "("))
                pos += 1
                continue
            if text[pos] == ")":
                self._tokens.append(("RPAREN", ")"))
                pos += 1
                continue

            # Check for bracket expressions (value paths)
            if text[pos] == "[":
                self._tokens.append(("LBRACKET", "["))
                pos += 1
                continue
            if text[pos] == "]":
                self._tokens.append(("RBRACKET", "]"))
                pos += 1
                continue

            # Check for string values
            match = re.match(self.STRING_VALUE, text[pos:])
            if match:
                value = match.group(0)[1:-1]  # Remove quotes
                self._tokens.append(("VALUE", value))
                pos += len(match.group(0))
                continue

            # Check for logical operators
            match = re.match(self.LOGICAL, text[pos:], re.IGNORECASE)
            if match:
                self._tokens.append(("LOGICAL", match.group(0).lower()))
                pos += len(match.group(0))
                continue

            # Check for comparison operators
            match = re.match(self.OPERATOR, text[pos:], re.IGNORECASE)
            if match:
                self._tokens.append(("OPERATOR", match.group(0).lower()))
                pos += len(match.group(0))
                continue

            # Check for boolean values
            match = re.match(self.BOOL_VALUE, text[pos:], re.IGNORECASE)
            if match:
                self._tokens.append(("VALUE", match.group(0).lower() == "true"))
                pos += len(match.group(0))
                continue

            # Check for number values
            match = re.match(self.NUMBER_VALUE, text[pos:])
            if match:
                num_str = match.group(0)
                num_value: float | int = float(num_str) if "." in num_str else int(num_str)
                self._tokens.append(("VALUE", num_value))
                pos += len(match.group(0))
                continue

            # Check for attributes
            match = re.match(self.ATTRIBUTE, text[pos:])
            if match:
                self._tokens.append(("ATTRIBUTE", match.group(0)))
                pos += len(match.group(0))
                continue

            # Unknown character
            raise ValueError(f"Invalid character in filter at position {pos}: {text[pos]}")

        self._pos = 0

    def _parse_expression(self) -> SCIMFilter | SCIMCompoundFilter:
        """Parse a filter expression."""
        left = self._parse_term()

        while self._pos < len(self._tokens):
            token_type, token_value = self._tokens[self._pos]

            if token_type == "LOGICAL" and token_value in ("and", "or"):  # noqa: S105 -- parser token tag
                self._pos += 1
                right = self._parse_term()
                left = SCIMCompoundFilter(
                    operator=SCIMLogicalOperator(token_value),
                    filters=[left, right],
                )
            else:
                break

        return left

    def _parse_term(self) -> SCIMFilter | SCIMCompoundFilter:
        """Parse a single filter term."""
        if self._pos >= len(self._tokens):
            raise ValueError("Unexpected end of filter expression")

        token_type, token_value = self._tokens[self._pos]  # noqa: S105 -- parser token tag

        # Handle NOT operator
        if token_type == "LOGICAL" and token_value == "not":  # noqa: S105 -- parser token tag
            self._pos += 1
            inner = self._parse_term()
            return SCIMCompoundFilter(
                operator=SCIMLogicalOperator.NOT,
                filters=[inner],
            )

        # Handle parentheses
        if token_type == "LPAREN":  # noqa: S105 -- parser token tag
            self._pos += 1
            expr = self._parse_expression()
            if self._pos >= len(self._tokens) or self._tokens[self._pos][0] != "RPAREN":
                raise ValueError("Missing closing parenthesis")
            self._pos += 1
            return expr

        # Parse attribute comparison
        if token_type == "ATTRIBUTE":  # noqa: S105 -- parser token tag
            attribute = token_value
            self._pos += 1

            # Handle value path (e.g., emails[type eq "work"])
            sub_attribute = None
            if self._pos < len(self._tokens) and self._tokens[self._pos][0] == "LBRACKET":
                self._pos += 1
                # Skip the sub-filter for now, just get to the attribute
                while self._pos < len(self._tokens) and self._tokens[self._pos][0] != "RBRACKET":
                    self._pos += 1
                if self._pos < len(self._tokens):
                    self._pos += 1  # Skip RBRACKET
                # Check for .value after bracket
                if self._pos < len(self._tokens) and self._tokens[self._pos][0] == "ATTRIBUTE":
                    if self._tokens[self._pos][1].startswith("."):
                        sub_attribute = self._tokens[self._pos][1][1:]
                        self._pos += 1

            if self._pos >= len(self._tokens):
                raise ValueError(f"Expected operator after attribute: {attribute}")

            op_type, op_value = self._tokens[self._pos]
            if op_type != "OPERATOR":
                raise ValueError(f"Expected operator, got: {op_value}")

            operator = SCIMOperator(op_value)
            self._pos += 1

            # Handle presence operator (no value needed)
            if operator == SCIMOperator.PR:
                return SCIMFilter(
                    attribute=attribute,
                    operator=operator,
                    value=None,
                    sub_attribute=sub_attribute,
                )

            # Get value
            if self._pos >= len(self._tokens):
                raise ValueError(f"Expected value after operator: {operator}")

            val_type, value = self._tokens[self._pos]
            if val_type != "VALUE":
                raise ValueError(f"Expected value, got: {value}")
            self._pos += 1

            return SCIMFilter(
                attribute=attribute,
                operator=operator,
                value=value,
                sub_attribute=sub_attribute,
            )

        raise ValueError(f"Unexpected token: {token_value}")


def parse_filter(filter_text: str) -> SCIMFilter | SCIMCompoundFilter | None:
    """
    Convenience function to parse a SCIM filter.

    Args:
        filter_text: The filter expression string

    Returns:
        A SCIMFilter or SCIMCompoundFilter object, or None if empty
    """
    return SCIMFilterParser().parse(filter_text)
