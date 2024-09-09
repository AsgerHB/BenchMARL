# Issues I've noticed.

### Custom Env

The [Example for creating custom environments](examples/extending/task/environments/customenv/common.py) sucks. There are so many little things with the dimensionality that is left ambiguous for a newbie.

### Callbacks

Apparently the [custom callbacks example](examples/callback/custom_callback.py) is wrong. The class functions should not include `self` in their definition.