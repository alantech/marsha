from __future__ import annotations

from typing import Literal, TypedDict

# The shape of one node in mistletoe's markdown AST, as produced by ast_renderer.get_ast (an
# untyped, but stable, third-party API). Each node is a plain dict whose `type` string names the
# kind of node; `type` is the discriminator, so mypy narrows the `AstNode` union on
# `node['type'] == 'X'`. Only the structurally meaningful fields are modeled (children, content,
# level, language, start, src/target/title); mistletoe's line-number and alignment metadata is
# omitted. The concrete node dicts carry extra keys, but marsha enters the type only through the
# cast in meta._get_ast, so that is not checked here.


class DocumentNode(TypedDict):
    type: Literal['Document']
    children: list[AstNode]


class HeadingNode(TypedDict):
    type: Literal['Heading']
    level: int
    children: list[AstNode]


class SetextHeadingNode(TypedDict):
    type: Literal['SetextHeading']
    level: int
    children: list[AstNode]


class ParagraphNode(TypedDict):
    type: Literal['Paragraph']
    children: list[AstNode]


class QuoteNode(TypedDict):
    type: Literal['Quote']
    children: list[AstNode]


class ListNode(TypedDict):
    type: Literal['List']
    start: int | None
    children: list[AstNode]


class ListItemNode(TypedDict):
    type: Literal['ListItem']
    children: list[AstNode]


class CodeFenceNode(TypedDict):
    type: Literal['CodeFence']
    language: str
    children: list[AstNode]


class BlockCodeNode(TypedDict):
    type: Literal['BlockCode']
    language: str
    children: list[AstNode]


class RawTextNode(TypedDict):
    type: Literal['RawText']
    content: str


class LineBreakNode(TypedDict):
    type: Literal['LineBreak']
    content: str


class ThematicBreakNode(TypedDict):
    type: Literal['ThematicBreak']


class EmphasisNode(TypedDict):
    type: Literal['Emphasis']
    children: list[AstNode]


class StrongNode(TypedDict):
    type: Literal['Strong']
    children: list[AstNode]


class InlineCodeNode(TypedDict):
    type: Literal['InlineCode']
    children: list[AstNode]


class StrikethroughNode(TypedDict):
    type: Literal['Strikethrough']
    children: list[AstNode]


class EscapeSequenceNode(TypedDict):
    type: Literal['EscapeSequence']
    children: list[AstNode]


class LinkNode(TypedDict):
    type: Literal['Link']
    target: str
    title: str
    children: list[AstNode]


class ImageNode(TypedDict):
    type: Literal['Image']
    src: str
    title: str
    children: list[AstNode]


class AutoLinkNode(TypedDict):
    type: Literal['AutoLink']
    children: list[AstNode]


# Table markup is not rendered by marsha (to_markdown raises NotImplementedError for these), so
# they are modeled as bare discriminants: just enough for the node union to be complete and for
# to_markdown's branches to be reachable.
class TableNode(TypedDict):
    type: Literal['Table']


class TableRowNode(TypedDict):
    type: Literal['TableRow']


class TableCellNode(TypedDict):
    type: Literal['TableCell']


type AstNode = (
    DocumentNode | HeadingNode | SetextHeadingNode | ParagraphNode | QuoteNode
    | ListNode | ListItemNode | CodeFenceNode | BlockCodeNode | RawTextNode
    | LineBreakNode | ThematicBreakNode | EmphasisNode | StrongNode | InlineCodeNode
    | StrikethroughNode | EscapeSequenceNode | LinkNode | ImageNode | AutoLinkNode
    | TableNode | TableRowNode | TableCellNode
)


def child_text(children: list[AstNode]) -> str:
    # The text of a single-leaf inline node (Emphasis/Strong/InlineCode/CodeFence/AutoLink/...):
    # its one RawText child. to_markdown handles only the simple single-child case for inline
    # containers (a nested inline node flattens to its first leaf), so the first child is
    # expected to be a RawText leaf; anything else is not the simple form to_markdown supports.
    first = children[0]
    if first['type'] == 'RawText':
        return first['content']
    raise Exception(f'expected a RawText leaf, got {first["type"]}')
