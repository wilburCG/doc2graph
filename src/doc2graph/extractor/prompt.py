"""LLM extraction prompt templates."""

EXTRACTION_SYSTEM_PROMPT = """你是一个知识图谱构建专家。请从文档中准确提取所有重要的实体和关系。

## 输出要求
- 提取所有关键实体（人物、公司、技术、概念、产品、事件等）
- 提取实体之间的关系
- 每个实体包含：名称(name)、类型(type)、描述(description)
- 每个关系包含：源实体(source)、目标实体(target)、关系类型(type)、描述(description)
- 关系类型用大写英文单词，如 CREATED、RELATED_TO、PART_OF、WORKS_FOR 等

## 实体类型参考
Person, Company, Organization, Technology, Product, Concept, Event, Location, Date

## 输出格式（严格 JSON）
{
  "entities": [
    {"name": "实体名", "type": "类型", "description": "描述"}
  ],
  "relations": [
    {"source": "源实体", "target": "目标实体", "type": "关系类型", "description": "描述"}
  ]
}"""

EXTRACTION_USER_TEMPLATE = """请从以下文档中提取实体和关系。

## 文档内容
{text}

## 约束
- 最多提取 {max_entities} 个实体
- 最多提取 {max_relations} 个关系
- 只输出 JSON，不要其他内容
- 确保 JSON 格式正确可解析"""
