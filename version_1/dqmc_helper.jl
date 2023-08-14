"""
Macro to define substitute macros.
Concretely, `@def mymacro body` will define a macro
such that `@mymacro` will be replaced by `body`.
"""
macro def(name, definition)
    return quote
        macro $(esc(name))()
            esc($(Expr(:quote, definition)))
        end
    end
end
export @def