---
jupyter:
  kernelspec:
    display_name: Python 3.10.4 64-bit
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.10.4
  nbformat: 4
  nbformat_minor: 4
  vscode:
    interpreter:
      hash: 36a0d8ae7ea53d0c2bb98a357b7e4d666f857d54e9cfe25daaf029ef2b61f7f5
---

::: {.cell .markdown}
# Paddy classification
:::

::: {.cell .code execution_count="8" execution="{\"iopub.status.busy\":\"2022-06-29T15:55:59.497464Z\",\"shell.execute_reply.started\":\"2022-06-29T15:55:59.498504Z\",\"iopub.status.idle\":\"2022-06-29T15:56:02.350146Z\",\"iopub.execute_input\":\"2022-06-29T15:55:59.498682Z\",\"shell.execute_reply\":\"2022-06-29T15:56:02.349128Z\"}" trusted="true"}
``` {.python}
from fastai.vision.all import *
import pandas as pd
%matplotlib inline
set_seed(3865)
```
:::

::: {.cell .code execution_count="9" execution="{\"iopub.status.busy\":\"2022-06-29T15:56:30.626091Z\",\"shell.execute_reply.started\":\"2022-06-29T15:56:30.626857Z\",\"iopub.status.idle\":\"2022-06-29T15:56:31.874625Z\",\"iopub.execute_input\":\"2022-06-29T15:56:30.626892Z\",\"shell.execute_reply\":\"2022-06-29T15:56:31.873683Z\"}" trusted="true"}
``` {.python}
import albumentations as Alb
class AlbTransform(Transform):
    def __init__(self, aug): self.aug = aug
    def encodes(self, img: PILImage):
        aug_img = self.aug(image=np.array(img))['image']
        return PILImage.create(aug_img)
    
def get_augs(): return Alb.Compose([
    Alb.ShiftScaleRotate(rotate_limit=20, border_mode=0, value=(0,0,0) ),
    Alb.Transpose(),
    Alb.Flip(),
    Alb.RandomRotate90(),
    Alb.RandomBrightnessContrast(),
    Alb.HueSaturationValue(
      hue_shift_limit=5, 
      sat_shift_limit=5, 
      val_shift_limit=5 ),
])

item_tfms = [Resize(224), AlbTransform(get_augs())] 
batch_tfms = Normalize.from_stats(*imagenet_stats) 
```
:::

::: {.cell .code execution_count="10" _kg_hide-input="true" execution="{\"iopub.status.busy\":\"2022-06-29T15:56:31.876219Z\",\"shell.execute_reply.started\":\"2022-06-29T15:56:31.876536Z\",\"iopub.status.idle\":\"2022-06-29T15:56:31.881987Z\",\"iopub.execute_input\":\"2022-06-29T15:56:31.876565Z\",\"shell.execute_reply\":\"2022-06-29T15:56:31.880538Z\"}" trusted="true"}
``` {.python}
# debug: mini-training

# !mkdir train
# !mkdir train/downy_mildew
# !mkdir train/normal
# !mkdir train/blast

# !cp ../input/paddy-disease-classification/train_images/downy_mildew/1000*.jpg train/downy_mildew
# !cp ../input/paddy-disease-classification/train_images/normal/1001*.jpg train/normal
# !cp ../input/paddy-disease-classification/train_images/blast/1001*.jpg train/blast
```
:::

::: {.cell .code execution_count="11" execution="{\"iopub.status.busy\":\"2022-06-29T15:56:31.883204Z\",\"iopub.execute_input\":\"2022-06-29T15:56:31.883617Z\"}" trusted="true"}
``` {.python}
# create dataloader from the folder structure
dls = ImageDataLoaders.from_folder( './paddy-disease-classification/train_images', 
    train='.', valid=None, valid_pct=0.01, 
    item_tfms=item_tfms, batch_tfms=batch_tfms, bs=32, shuffle=True )
```

::: {.output .stream .stdout}
    Due to IPython and Windows limitation, python multiprocessing isn't available now.
    So `number_workers` is changed to 0 to avoid getting stuck
:::
:::

::: {.cell .code execution_count="12" trusted="true"}
``` {.python}
# uncomment to test data loaders
# dls.train.show_batch(max_n=12)
print('train items:', len(dls.train.items), 'validation items:', len(dls.valid.items))
# dls.valid.show_batch(max_n=12)
dls.vocab
```

::: {.output .stream .stdout}
    train items: 10303 validation items: 104
:::

::: {.output .execute_result execution_count="12"}
    ['bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight', 'blast', 'brown_spot', 'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro']
:::
:::

::: {.cell .code execution_count="13" trusted="true"}
``` {.python}
# learn = vision_learner(dls, densenet121, path='.', 
learn = vision_learner(dls, resnet50, path='.', 
    loss_func=FocalLoss(),  
    metrics=[accuracy]  )
```

::: {.output .stream .stderr}
    Downloading: "https://download.pytorch.org/models/resnet50-0676ba61.pth" to C:\Users\rohit/.cache\torch\hub\checkpoints\resnet50-0676ba61.pth
:::

::: {.output .display_data}
``` {.json}
{"version_major":2,"version_minor":0,"model_id":"f290cf45b6f248398ae5d3233236f730"}
```
:::
:::

::: {.cell .code execution_count="14" trusted="true"}
``` {.python}
learn.fine_tune(30, freeze_epochs=1, cbs=[ShowGraphCallback()])
```

::: {.output .display_data}
```{=html}
<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>
```
:::

::: {.output .display_data}
```{=html}
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.410440</td>
      <td>0.751294</td>
      <td>0.682692</td>
      <td>58:51</td>
    </tr>
  </tbody>
</table>
```
:::

::: {.output .display_data}
![](vertopal_baa77b392ec74855be4a5d0d793c182c/053cefab270491e29d8d90448aa7ac2b23d0c026.png)
:::

::: {.output .display_data}
```{=html}
<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>
```
:::

::: {.output .display_data}
```{=html}
    <div>
      <progress value='2' class='' max='30' style='width:300px; height:20px; vertical-align: middle;'></progress>
      6.67% [2/30 13:41:08<191:35:53]
    </div>
    
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.749519</td>
      <td>0.527632</td>
      <td>0.750000</td>
      <td>1:16:33</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.501818</td>
      <td>0.344970</td>
      <td>0.875000</td>
      <td>12:24:32</td>
    </tr>
  </tbody>
</table><p>

    <div>
      <progress value='17' class='' max='321' style='width:300px; height:20px; vertical-align: middle;'></progress>
      5.30% [17/321 05:39<1:41:06 0.4879]
    </div>
    
```
:::

::: {.output .display_data}
![](vertopal_baa77b392ec74855be4a5d0d793c182c/e7f17b93a218dbcada35ca548ba710a3f519b446.png)
:::

::: {.output .error ename="KeyboardInterrupt" evalue=""}
    ---------------------------------------------------------------------------
    KeyboardInterrupt                         Traceback (most recent call last)
    d:\python\paddyDiseases\paddydiseases-fastai2.ipynb Cell 8' in <cell line: 1>()
    ----> <a href='vscode-notebook-cell:/d%3A/python/paddyDiseases/paddydiseases-fastai2.ipynb#ch0000007?line=0'>1</a> learn.fine_tune(30, freeze_epochs=1, cbs=[ShowGraphCallback()])

    File c:\Users\rohit\AppData\Local\Programs\Python\Python310\lib\site-packages\fastai\callback\schedule.py:171, in fine_tune(self, epochs, base_lr, freeze_epochs, lr_mult, pct_start, div, **kwargs)
        169 base_lr /= 2
        170 self.unfreeze()
    --> 171 self.fit_one_cycle(epochs, slice(base_lr/lr_mult, base_lr), pct_start=pct_start, div=div, **kwargs)

    File c:\Users\rohit\AppData\Local\Programs\Python\Python310\lib\site-packages\fastai\callback\schedule.py:122, in fit_one_cycle(self, n_epoch, lr_max, div, div_final, pct_start, wd, moms, cbs, reset_opt, start_epoch)
        119 lr_max = np.array([h['lr'] for h in self.opt.hypers])
        120 scheds = {'lr': combined_cos(pct_start, lr_max/div, lr_max, lr_max/div_final),
        121           'mom': combined_cos(pct_start, *(self.moms if moms is None else moms))}
    --> 122 self.fit(n_epoch, cbs=ParamScheduler(scheds)+L(cbs), reset_opt=reset_opt, wd=wd, start_epoch=start_epoch)

    File c:\Users\rohit\AppData\Local\Programs\Python\Python310\lib\site-packages\fastai\learner.py:241, in Learner.fit(self, n_epoch, lr, wd, cbs, reset_opt, start_epoch)
        239 self.opt.set_hypers(lr=self.lr if lr is None else lr)
        240 self.n_epoch = n_epoch
    --> 241 self._with_events(self._do_fit, 'fit', CancelFitException, self._end_cleanup)

    File c:\Users\rohit\AppData\Local\Programs\Python\Python310\lib\site-packages\fastai\learner.py:179, in Learner._with_events(self, f, event_type, ex, final)
        178 def _with_events(self, f, event_type, ex, final=noop):
    --> 179     try: self(f'before_{event_type}');  f()
        180     except ex: self(f'after_cancel_{event_type}')
        181     self(f'after_{event_type}');  final()

    File c:\Users\rohit\AppData\Local\Programs\Python\Python310\lib\site-packages\fastai\learner.py:230, in Learner._do_fit(self)
        228 for epoch in range(self.n_epoch):
        229     self.epoch=epoch
    --> 230     self._with_events(self._do_epoch, 'epoch', CancelEpochException)

    File c:\Users\rohit\AppData\Local\Programs\Python\Python310\lib\site-packages\fastai\learner.py:179, in Learner._with_events(self, f, event_type, ex, final)
        178 def _with_events(self, f, event_type, ex, final=noop):
    --> 179     try: self(f'before_{event_type}');  f()
        180     except ex: self(f'after_cancel_{event_type}')
        181     self(f'after_{event_type}');  final()

    File c:\Users\rohit\AppData\Local\Programs\Python\Python310\lib\site-packages\fastai\learner.py:224, in Learner._do_epoch(self)
        223 def _do_epoch(self):
    --> 224     self._do_epoch_train()
        225     self._do_epoch_validate()

    File c:\Users\rohit\AppData\Local\Programs\Python\Python310\lib\site-packages\fastai\learner.py:216, in Learner._do_epoch_train(self)
        214 def _do_epoch_train(self):
        215     self.dl = self.dls.train
    --> 216     self._with_events(self.all_batches, 'train', CancelTrainException)

    File c:\Users\rohit\AppData\Local\Programs\Python\Python310\lib\site-packages\fastai\learner.py:179, in Learner._with_events(self, f, event_type, ex, final)
        178 def _with_events(self, f, event_type, ex, final=noop):
    --> 179     try: self(f'before_{event_type}');  f()
        180     except ex: self(f'after_cancel_{event_type}')
        181     self(f'after_{event_type}');  final()

    File c:\Users\rohit\AppData\Local\Programs\Python\Python310\lib\site-packages\fastai\learner.py:185, in Learner.all_batches(self)
        183 def all_batches(self):
        184     self.n_iter = len(self.dl)
    --> 185     for o in enumerate(self.dl): self.one_batch(*o)

    File c:\Users\rohit\AppData\Local\Programs\Python\Python310\lib\site-packages\fastai\learner.py:212, in Learner.one_batch(self, i, b)
        210 b = self._set_device(b)
        211 self._split(b)
    --> 212 self._with_events(self._do_one_batch, 'batch', CancelBatchException)

    File c:\Users\rohit\AppData\Local\Programs\Python\Python310\lib\site-packages\fastai\learner.py:179, in Learner._with_events(self, f, event_type, ex, final)
        178 def _with_events(self, f, event_type, ex, final=noop):
    --> 179     try: self(f'before_{event_type}');  f()
        180     except ex: self(f'after_cancel_{event_type}')
        181     self(f'after_{event_type}');  final()

    File c:\Users\rohit\AppData\Local\Programs\Python\Python310\lib\site-packages\fastai\learner.py:198, in Learner._do_one_batch(self)
        196 self('after_loss')
        197 if not self.training or not len(self.yb): return
    --> 198 self._with_events(self._backward, 'backward', CancelBackwardException)
        199 self._with_events(self._step, 'step', CancelStepException)
        200 self.opt.zero_grad()

    File c:\Users\rohit\AppData\Local\Programs\Python\Python310\lib\site-packages\fastai\learner.py:179, in Learner._with_events(self, f, event_type, ex, final)
        178 def _with_events(self, f, event_type, ex, final=noop):
    --> 179     try: self(f'before_{event_type}');  f()
        180     except ex: self(f'after_cancel_{event_type}')
        181     self(f'after_{event_type}');  final()

    File c:\Users\rohit\AppData\Local\Programs\Python\Python310\lib\site-packages\fastai\learner.py:187, in Learner._backward(self)
    --> 187 def _backward(self): self.loss_grad.backward()

    File c:\Users\rohit\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\_tensor.py:355, in Tensor.backward(self, gradient, retain_graph, create_graph, inputs)
        308 r"""Computes the gradient of current tensor w.r.t. graph leaves.
        309 
        310 The graph is differentiated using the chain rule. If the tensor is
       (...)
        352         used to compute the attr::tensors.
        353 """
        354 if has_torch_function_unary(self):
    --> 355     return handle_torch_function(
        356         Tensor.backward,
        357         (self,),
        358         self,
        359         gradient=gradient,
        360         retain_graph=retain_graph,
        361         create_graph=create_graph,
        362         inputs=inputs)
        363 torch.autograd.backward(self, gradient, retain_graph, create_graph, inputs=inputs)

    File c:\Users\rohit\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\overrides.py:1394, in handle_torch_function(public_api, relevant_args, *args, **kwargs)
       1388     warnings.warn("Defining your `__torch_function__ as a plain method is deprecated and "
       1389                   "will be an error in PyTorch 1.11, please define it as a classmethod.",
       1390                   DeprecationWarning)
       1392 # Use `public_api` instead of `implementation` so __torch_function__
       1393 # implementations can do equality/identity comparisons.
    -> 1394 result = torch_func_method(public_api, types, args, kwargs)
       1396 if result is not NotImplemented:
       1397     return result

    File c:\Users\rohit\AppData\Local\Programs\Python\Python310\lib\site-packages\fastai\torch_core.py:365, in TensorBase.__torch_function__(cls, func, types, args, kwargs)
        363 if cls.debug and func.__name__ not in ('__str__','__repr__'): print(func, types, args, kwargs)
        364 if _torch_handled(args, cls._opt, func): types = (torch.Tensor,)
    --> 365 res = super().__torch_function__(func, types, args, ifnone(kwargs, {}))
        366 dict_objs = _find_args(args) if args else _find_args(list(kwargs.values()))
        367 if issubclass(type(res),TensorBase) and dict_objs: res.set_meta(dict_objs[0],as_copy=True)

    File c:\Users\rohit\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\_tensor.py:1142, in Tensor.__torch_function__(cls, func, types, args, kwargs)
       1139     return NotImplemented
       1141 with _C.DisableTorchFunction():
    -> 1142     ret = func(*args, **kwargs)
       1143     if func in get_default_nowrap_functions():
       1144         return ret

    File c:\Users\rohit\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\_tensor.py:363, in Tensor.backward(self, gradient, retain_graph, create_graph, inputs)
        354 if has_torch_function_unary(self):
        355     return handle_torch_function(
        356         Tensor.backward,
        357         (self,),
       (...)
        361         create_graph=create_graph,
        362         inputs=inputs)
    --> 363 torch.autograd.backward(self, gradient, retain_graph, create_graph, inputs=inputs)

    File c:\Users\rohit\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\autograd\__init__.py:173, in backward(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)
        168     retain_graph = create_graph
        170 # The reason we repeat same the comment below is that
        171 # some Python versions print out the first line of a multi-line function
        172 # calls in the traceback and some print out the last line
    --> 173 Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
        174     tensors, grad_tensors_, retain_graph, create_graph, inputs,
        175     allow_unreachable=True, accumulate_grad=True)

    KeyboardInterrupt: 
:::

::: {.output .display_data}
![](vertopal_baa77b392ec74855be4a5d0d793c182c/e7f17b93a218dbcada35ca548ba710a3f519b446.png)
:::
:::

::: {.cell .code trusted="true"}
``` {.python}
# learn.freeze()
# learn.fit_one_cycle(1)
# learn.unfreeze()
```
:::

::: {.cell .code trusted="true"}
``` {.python}
# import time
# startTime = time.time()

# lrs = learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))
# print('lr_find in:', time.time()-startTime, 'secs', 'lrs.valley=', lrs.valley)
# learn.fit_one_cycle(21, lr_max=lrs.valley, wd=0.2, cbs=[ShowGraphCallback()])
```
:::

::: {.cell .code trusted="true"}
``` {.python}
# prepare test data
ftest = get_image_files('../input/paddy-disease-classification/test_images')
# ftest = ftest[:50] # scale down for debug
print('Testing', len(ftest), 'items')
```
:::

::: {.cell .code trusted="true"}
``` {.python}
# make dataloader for test data
tst_dl = dls.test_dl(ftest, with_labels=False, shuffle=False)
# uncomment to see if dataloader is working
tst_dl.show_batch(max_n=12)
```
:::

::: {.cell .code trusted="true"}
``` {.python}
startTime = time.time()
preds = learn.tta(dl=tst_dl, n=32, use_max=False)
print('TTA in:', time.time()-startTime, 'secs')
```
:::

::: {.cell .code trusted="true"}
``` {.python}
predss = learn.dls.vocab[np.argmax(preds[0], axis=1)] # convert to our classes from probabilities
subm_df = pd.DataFrame()
subm_df['image_id'] = [item.name for item in tst_dl.items]
subm_df['label'] = predss
subm_df.to_csv('submission.csv', header=True, index=False)
```
:::

::: {.cell .code trusted="true"}
``` {.python}
subm_df
```
:::

::: {.cell .markdown}
# Have a nice day!
:::

::: {.cell .code}
``` {.python}
```
:::
