# Business logic / services package
# Services contain the actual logic (not just routing).
# Example: ReportService.generate() would:
#   1. Retrieve context from Qdrant
#   2. Build a prompt
#   3. Call the LLM
#   4. Format the response
# This separation keeps endpoints thin and logic testable.
