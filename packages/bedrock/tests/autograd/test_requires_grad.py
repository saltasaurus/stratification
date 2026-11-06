import pytest
from bedrock.autograd.tensaur import Tensaur


@pytest.mark.parametrize("train_left", [True, False])
def test_add_only_accumulates_into_enabled_input(train_left):
    a = Tensaur(2.0, requires_grad=train_left)
    b = Tensaur(3.0, requires_grad=not train_left)

    out = a + b
    out.backward()

    assert out.data == pytest.approx(5.0)
    assert out.requires_grad is True
    assert a.grad == pytest.approx(1.0 if train_left else 0.0)
    assert b.grad == pytest.approx(0.0 if train_left else 1.0)


@pytest.mark.parametrize("train_left", [True, False])
def test_mul_only_accumulates_into_enabled_input(train_left):
    a = Tensaur(2.0, requires_grad=train_left)
    b = Tensaur(3.0, requires_grad=not train_left)

    out = a * b
    out.backward()

    assert out.data == pytest.approx(6.0)
    assert out.requires_grad is True
    assert a.grad == pytest.approx(3.0 if train_left else 0.0)
    assert b.grad == pytest.approx(0.0 if train_left else 2.0)
