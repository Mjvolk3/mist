---
id: 7s0iisntw9xi81ep5qnr7i1
title: FormulaTransformer
desc: ''
updated: 1682827161633
created: 1682827125411
---
## FormulaEncoder Options

1. When `self.single_form_encoder` is True:

   - The formula dimension (`self.formula_dim`) is set as the sum of `utils.ELEMENT_DIM_MASS` and `self.num_types`.
   - A single `dense_encoder` is created as an instance of `nn.Sequential`, which is a linear container of layers, including two Linear layers with ReLU activation functions and dropout layers in between.
   - The `dense_encoder` is then wrapped inside an `nn.ModuleList`, which allows it to be treated as a single module with its parameters registered for optimization.
   - The `self.formula_encoders` is assigned this single-module `nn.ModuleList` containing the `dense_encoder`.
   - `self.onehot_types` is assigned the `nn` (PyTorch's `torch.nn` module), but this assignment doesn't seem to be used anywhere else in the code snippet and is likely not necessary.

2. When `self.single_form_encoder` is False:

   - The formula dimension (`self.formula_dim`) is set as `utils.ELEMENT_DIM_MASS`.
   - A list `dense_formula_encoders` is initialized as an empty list.
   - For each type in `range(self.num_types)`, a new `dense_encoder` is created, which has the same architecture as the one described in the first condition (two Linear layers with ReLU activation functions and dropout layers in between).
   - Each of these `dense_encoder` instances is appended to the `dense_formula_encoders` list.
   - The `self.formula_encoders` is assigned an `nn.ModuleList` containing all the `dense_encoder` instances in the `dense_formula_encoders` list.

In summary, the key difference between the two options is:

- When `self.single_form_encoder` is True, a single shared `dense_encoder` is created and used for all types.
- When `self.single_form_encoder` is False, separate `dense_encoder` instances are created for each type, which means that each type has its own distinct formula encoder.

The choice between these two options depends on the desired behavior of the model. If you want to share the same formula encoder among all types, you would set `self.single_form_encoder` to True. On the other hand, if you want separate formula encoders for each type, you would set `self.single_form_encoder` to False.
