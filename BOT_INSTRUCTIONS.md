# Bot Instructions

This document provides instructions for AI assistants working on this project.

## Project Status Review

When asked to review project status, check these files in order:

1. `CHANGELOG.md`
   - Most recent changes
   - Current version
   - Unreleased features

2. `MODEL_TYPES.md`
   - Implementation status of features
   - Supported model types
   - Current priorities

3. `README.md`
   - Completed features
   - In-progress features
   - Upcoming features

4. `PROJECT_STRUCTURE.md`
   - Current architecture
   - Component descriptions
   - Development status

## Making Changes

When implementing new features or making changes:

1. Update Documentation:
   - Add new changes to `CHANGELOG.md` under [Unreleased]
   - Update implementation status in `MODEL_TYPES.md`
   - Update features list in `README.md`
   - Update architecture in `PROJECT_STRUCTURE.md` if needed

2. Update Tests:
   - Add/modify tests in `tests/` directory
   - Ensure test documentation is updated

3. Update Code:
   - Follow existing patterns and style
   - Add proper documentation
   - Include type hints

## Standard Response Format

When continuing work, provide:

1. Status Overview:
   ```
   Current Status:
   - Last completed: [feature/change]
   - In progress: [feature/change]
   - Next up: [feature/change]
   ```

2. Proposed Changes:
   ```
   Proposed Implementation:
   - Files to modify
   - Changes to make
   - Tests to add/update
   ```

3. Documentation Updates:
   ```
   Documentation Updates:
   - CHANGELOG.md: [changes]
   - MODEL_TYPES.md: [changes]
   - README.md: [changes]
   - PROJECT_STRUCTURE.md: [changes]
   ```

## Example Response

```
Reviewing project status...

Current Status:
- Last completed: Universal model conversion system
- In progress: Testing framework implementation
- Next up: CLI interface development

Proposed Implementation:
- Create cli.py interface
- Add argument parsing
- Implement conversion commands
- Add progress reporting

Documentation Updates:
- CHANGELOG.md: Add CLI interface implementation
- MODEL_TYPES.md: Update testing status
- README.md: Add CLI features
- PROJECT_STRUCTURE.md: Add CLI component

Would you like me to proceed with the proposed changes?
```

## Project Priorities

Always prioritize in this order:

1. Documentation accuracy and consistency
2. Core functionality
3. Testing coverage
4. User experience improvements
5. Performance optimizations

## Version Control

Track semantic versioning:
- MAJOR: Breaking changes
- MINOR: New features
- PATCH: Bug fixes

## Error Handling

If encountering inconsistencies:
1. Point them out
2. Propose fixes
3. Wait for confirmation before making changes

## Best Practices

1. Keep all documentation files in sync
2. Use consistent formatting
3. Follow existing patterns
4. Add comprehensive comments
5. Maintain test coverage