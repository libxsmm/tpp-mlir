# RUN: python3 %s | FileCheck %s

import mlir.ir as ir
from mlir.dialects import transform, func, linalg, tensor
from mlir.dialects.transform import interpreter as transform_interpreter

from mlir.tpp.sched.common import match, call_with, callback


def gen_payload():
    payload_mod = ir.Module.create()
    with ir.InsertionPoint(payload_mod.body):
        matrix2k2k = ir.RankedTensorType.get([2048, 2048], ir.F32Type.get())

        @func.func(matrix2k2k, matrix2k2k, matrix2k2k)
        def payload(A, B, C):
            return (linalg.matmul(A, B, outs=(C,)),)

    return payload_mod


def gen_schedule(f):
    schedule = ir.Module.create()
    schedule.operation.attributes["transform.with_named_sequence"] = ir.UnitAttr.get()
    with ir.InsertionPoint(schedule.body):
        named_sequence = transform.NamedSequenceOp(
            "__transform_main",
            [transform.AnyOpType.get()],  # input types
            [],  # output types
            arg_attrs=[{"transform.readonly": ir.UnitAttr.get()}],
        )
        with ir.InsertionPoint(named_sequence.body):
            f(named_sequence.bodyTarget)
            # The main handle that each of the bundles operates on is to the
            # module which directly contains the `func.func`s.
            transform.yield_(())

    return schedule


def run_schedule(f):
    print("\n// TEST:", f.__name__, flush=True)
    with ir.Context(), ir.Location.unknown():
        container = ir.Module.create()
        container.body.append((schedule := gen_schedule(f)).operation)
        container.body.append((payload := gen_payload()).operation)
        transform_interpreter.apply_named_sequence(
            payload, schedule.body.operations[0], container
        )
        print(container, flush=True)
        del payload
        del schedule
    return f


# CHECK-LABEL: TEST: testTransformCallbackAttrAnnotation
# CHECK: transform.named_sequence
# CHECK: transform.tune.callback @called_directly(
# CHECK: transform.tune.callback @called_multiple_times(
# CHECK: transform.tune.callback @called_multiple_times(
# CHECK: func.func @payload
# CHECK-SAME: direct = "A"
# CHECK-SAME: indirect = "B"
# CHECK: linalg.matmul
# CHECK-NOT: direct = "A"
# CHECK-SAME: indirect = "B"
@run_schedule
def testTransformCallbackAttrAnnotation(target):
    func_ops = match(target, ops={"func.func"})

    @call_with(func_ops)
    def called_directly(funcs):
        for func in funcs:
            func.attributes["direct"] = ir.StringAttr.get("A")

    @callback
    def called_multiple_times(funcs):
        for func in funcs:
            func.attributes["indirect"] = ir.StringAttr.get("B")

    called_multiple_times(func_ops)
    matmul_ops = match(target, ops={"linalg.matmul"})
    called_multiple_times(matmul_ops)


# CHECK-LABEL: TEST: testTransformCallbackPyTransform
# CHECK: IR printer: indirectly obtained matmul
# CHECK: linalg.matmul
# CHECK: transform.named_sequence
# CHECK: %[[FUNCS:.*]] = transform.structured.match ops{["func.func"]}
# CHECK: %[[MATMULS:.*]] = transform.structured.match ops{["linalg.matmul"]}
# CHECK: %[[PARAM:.*]] = transform.param.constant
# CHECK: %[[RES:.*]] = transform.tune.callback @py_transform(%[[FUNCS]], %[[MATMULS]], %[[PARAM]])
# CHECK-SAME: : (!transform.any_op, !transform.any_op, !transform.any_param) -> !transform.any_value
# CHECK: func.func @payload
# CHECK-SAME: marked_with = "ABC"
# CHECK-NOT: linalg.matmul
# CHECK: tensor.empty
# CHECK-NOT: linalg.matmul
# CHECK: return
@run_schedule
def testTransformCallbackPyTransform(target):
    func_ops = match(target, ops={"func.func"})
    matmul_ops = match(target, ops={"linalg.matmul"})
    td_param = transform.param_constant(
        transform.AnyParamType.get(), ir.StringAttr.get("ABC")
    )

    @call_with(func_ops, matmul_ops, td_param)
    def py_transform(funcs, matmuls, params) -> (transform.AnyValueType.get(),):
        for func, param in zip(funcs, params):
            func.attributes["marked_with"] = param
        with ir.InsertionPoint(matmuls[0]):
            empty = tensor.empty([2048, 2048], ir.F32Type.get())
            indirectly_obtained_return = next(matmuls[0].result.uses).owner
            indirectly_obtained_return.operands[0] = empty
        return [[matmul.result for matmul in matmuls]]

    matmuls_result_handle = py_transform.result
    indirectly_obtained_matmul = transform.get_defining_op(
        transform.AnyOpType.get(), matmuls_result_handle
    )
    transform.print_(
        target=indirectly_obtained_matmul, name="indirectly obtained matmul"
    )
    transform.ApplyDeadCodeEliminationOp(
        func_ops
    )  # Cleans up matmul as it no longer has users.
