"""
Quick test to verify AgentOrchestrator works with agent lists
"""

from agent_toolkit import Agent, AgentOrchestrator


class MockAgent(Agent):
    """Simple mock agent for testing"""
    def build_prompts(self, context):
        return "system", "user"
    
    def parse_response(self, raw):
        return {"test": "output", "approved": True}
    
    def __call__(self, context):
        # Override to avoid LLM calls
        return {
            "agent": self.name,
            "raw": f"Mock response from {self.name}",
            "parsed": {"test": f"output from {self.name}", "approved": True}
        }


# Test 1: Two-agent workflow (maintains 'reasoning'/'validation' keys)
print("Test 1: Two-agent workflow")
reasoning = MockAgent(name="reasoning", system_prompt_template="test")
validation = MockAgent(name="validation", system_prompt_template="test")
orchestrator_two = AgentOrchestrator(agents=[reasoning, validation], save_dir=None)

context = {"thematic_object": "test", "object_description": "test", "base_polygon_name": "test"}
results = orchestrator_two.run_rounds(context, num_rounds=1)

print(f"  Agents: {[a.name for a in orchestrator_two.agents]}")
print(f"  Round keys: {list(results[0].keys())}")
assert "reasoning" in results[0], "Two-agent mode should have 'reasoning' key"
assert "validation" in results[0], "Two-agent mode should have 'validation' key"
print("  ✓ Two-agent workflow works\n")


# Test 2: Multi-agent workflow (3+ agents)
print("Test 2: Multi-agent workflow (3 agents)")
agent1 = MockAgent(name="analyzer", system_prompt_template="test")
agent2 = MockAgent(name="reviewer", system_prompt_template="test")
agent3 = MockAgent(name="optimizer", system_prompt_template="test")
orchestrator_multi = AgentOrchestrator(agents=[agent1, agent2, agent3], save_dir=None)

results_multi = orchestrator_multi.run_rounds(context, num_rounds=1)

print(f"  Agents: {[a.name for a in orchestrator_multi.agents]}")
print(f"  Round keys: {list(results_multi[0].keys())}")
assert "analyzer" in results_multi[0], "Multi-agent mode should have 'analyzer' key"
assert "reviewer" in results_multi[0], "Multi-agent mode should have 'reviewer' key"
assert "optimizer" in results_multi[0], "Multi-agent mode should have 'optimizer' key"
print("  ✓ Multi-agent workflow works\n")


print("="*60)
print("All tests passed! AgentOrchestrator works correctly.")
print("="*60)
