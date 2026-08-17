{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :inherited-members:
   :exclude-members: __init__, __new__

   {% block methods %}
   {% set public_methods = methods | reject("in", inherited_members) | reject("eq", "__init__") | list %}
   {% if public_methods %}
   .. rubric:: Methods

   .. autosummary::
   {% for item in public_methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
