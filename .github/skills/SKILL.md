# Agent Skills Guide for MEssE

This guide helps you leverage AI agents (like GitHub Copilot) to accelerate development on the Model Essence Extractor (MEssE) project.

## 🎯 Quick Start Commands

### Code Understanding
```
"Explain how the ComIn plugin interfaces with ICON"
"Show me the data flow from ICON to the ML model"
"What does the YAC coupling library do?"
"Walk me through the build process step by step"
```

### Debugging
```
"Check all Python scripts for syntax errors"
"Find why the ICON build is failing"
"Debug the SLURM job submission issues"
"Analyze error logs in the terminal output"
```

### Documentation
```
"Fill in all empty README files"
"Document the comin_plugin.py module"
"Create API documentation for the plugin functions"
"Write usage examples for prepare_data.sh"
```

### Code Improvement
```
"Add error handling to build_icon.sh"
"Refactor the plugin code for better readability"
"Add input validation to the Python scripts"
"Optimize the SLURM resource allocation"
```

### Testing
```
"Create unit tests for the ComIn plugin"
"Write integration tests for the ICON-ML coupling"
"Add validation checks for input data formats"
"Generate test data for the plugin"
```

### Search & Analysis
```
"Find all hardcoded paths in the codebase"
"List all environment variables used"
"Show me all TODO and FIXME comments"
"Find where the namelist parameters are defined"
```

## 📋 Common Workflows

### 1. Setting Up New Environment
**Task**: Help a new collaborator get started

**Prompt**:
```
"Create a comprehensive QUICKSTART.md that guides someone through:
1. Cloning the repository
2. Setting up the environment on Levante
3. Building all components (ICON, ComIn, YAC)
4. Running their first experiment"
```

### 2. Adding New Features
**Task**: Implement a new ML algorithm in the plugin

**Prompt**:
```
"Help me add a new neural network architecture to comin_plugin.py:
1. Review the current code structure
2. Show me where to add the new model class
3. Update the configuration to support the new model
4. Add tests for the new functionality"
```

### 3. Debugging Build Issues
**Task**: ICON compilation fails

**Prompt**:
```
"I'm getting a build error. Here's the error message: [paste error].
1. Analyze what's causing the issue
2. Check the build script for problems
3. Suggest fixes
4. Update the build script if needed"
```

### 4. Optimizing Performance
**Task**: Speed up the coupling workflow

**Prompt**:
```
"Analyze the performance bottlenecks in our ICON-ML coupling:
1. Review the data exchange patterns in comin_plugin.py
2. Check the SLURM job configuration for optimization opportunities
3. Suggest parallelization strategies
4. Implement the improvements"
```

### 5. Code Review & Quality
**Task**: Prepare code for open source release

**Prompt**:
```
"Audit the entire codebase for:
1. Missing or incomplete documentation
2. Hardcoded paths that should be configurable
3. Error handling gaps
4. Code style inconsistencies
Then fix the top 10 issues"
```

## 🔧 Advanced Techniques

### Multi-Step Tasks with Planning
For complex tasks, let the agent break them down:

```
"I need to add checkpoint/restart functionality to the plugin.
Please:
1. Research the current state handling
2. Design a checkpoint system
3. Implement save/load functions
4. Add tests
5. Update documentation"
```

### Parallel Research
Use the agent to gather information efficiently:

```
"Compare our ComIn implementation with standard coupling approaches.
Find examples from other projects and suggest improvements."
```

### Contextual Analysis
Let the agent understand relationships:

```
"How do changes in icon_master.namelist affect the plugin behavior?
Trace the data flow from namelist to Python."
```

### Automated Refactoring
Improve code systematically:

```
"Refactor all build scripts to:
1. Use consistent error handling
2. Add progress logging
3. Support dry-run mode
4. Validate prerequisites before running"
```

## 💡 Best Practices

### Be Specific
❌ "Fix the build script"
✅ "Add error checking to build_icon.sh that validates Spack is loaded before attempting to build"

### Provide Context
❌ "This doesn't work"
✅ "When I run build_ComIn.sh, it fails at the cmake step with error [paste error]. I'm on Levante and have loaded the environment using activate_levante_env."

### Iterate Gradually
❌ "Rewrite everything to be better"
✅ "First, let's add logging to build_icon.sh, then we'll improve error handling, then add configuration validation"

### Verify Results
❌ Accept changes blindly
✅ "Can you explain what this change does? Let's test it on a small example first."

## 🎓 Learning Progression

### Beginner
1. Ask for explanations of existing code
2. Request documentation improvements
3. Get help with debugging specific errors

### Intermediate
1. Request code refactoring and optimization
2. Add new features with guided assistance
3. Create comprehensive test suites

### Advanced
1. Design new architectures with agent collaboration
2. Perform large-scale code reorganization
3. Optimize performance across the entire workflow

## 🚀 Project-Specific Tips

### For ICON Development
```
"Check if my namelist configuration is compatible with ICON 2.6.6"
"Generate a new experiment configuration based on the master namelist"
"Explain the physics parameterizations in NAMELIST_ICON"
```

### For ComIn/YAC Integration
```
"Review the coupling strategy in comin_plugin.py"
"Add support for additional ICON variables in the coupling"
"Debug the YAC field exchange timing"
```

### For ML Components
```
"Implement a new loss function in the plugin"
"Add data augmentation to the training pipeline"
"Profile the ML inference performance during coupled runs"
```

### For HPC/Levante
```
"Optimize the SLURM script for better resource utilization"
"Add node-level performance monitoring"
"Configure the job array for parameter sweeps"
```

## 📚 Resources

- Main README: `/README.md`
- Build Scripts: `/scripts/build_scripts/`
- Plugin Code: `/scripts/plugin/`
- Experiment Config: `/experiment/`
- Meeting Notes: `/meeting_minutes.md`

## 🤝 Contributing

When using agent assistance for contributions:
1. Review all generated code carefully
2. Test changes before committing
3. Document any agent-assisted modifications
4. Share useful prompts with the team

---

**Remember**: The agent is a powerful assistant, but you're still the expert on your project. Use it to accelerate your work, not replace your judgment.

For questions or to share effective prompts, update this guide via pull request!
