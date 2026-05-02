import abc
from typing import Dict


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict) -> Dict:
        """Infer actions from observations."""

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        pass

    def get_prefix_rep(self, obs: Dict, *, last_rep: bool = False) -> Dict:
        """Get the prefix representation (VLM hidden states) for the given observation.

        This is useful for RL or downstream tasks that need access to the
        VLM prefix representations without running the full diffusion loop.

        Args:
            obs: The observation dictionary.
            last_rep: If True, return only the last token's representation [B, W]
                      instead of the full sequence [B, S_prefix, W].
        """
        raise NotImplementedError("get_prefix_rep is not implemented for this policy.")
