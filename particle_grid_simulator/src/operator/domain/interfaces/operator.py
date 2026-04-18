from typing import Protocol, Type, Callable, Any, TypeVar

T = TypeVar('T')


class IOperatorData(Protocol[T]):

    @property
    def evolution_function(self) -> Callable[[Any], T]:
        """
        The mathematical rule (Theta).
        Defined as a property to ensure read-only access for safe kernel extraction.
        """
        ...

    @property
    def state_class_ref(self) -> Type[T]:
        """
        The guarantee of what this rule produces (S).
        Defined as a property to prevent runtime mutation of the return contract.
        """
        ...


class IOperatorUtility(Protocol):

    @staticmethod
    def evolve(operator: IOperatorData[T], data_context: Any) -> T:
        """
                Evaluates the operator's rule against a given context.

                Args:
                    operator: Any object fulfilling the IOperatorData contract.
                    data_context: The parametric space (Omega) containing states/fields.

                Returns:
                    An instance of the state class defined in the operator.
        """
        ...