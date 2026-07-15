import re

def format_latex(text):
    if not text:
        return text
    text = str(text)
    
    is_latex = '$' in text
    
    # Remove math mode $
    text = text.replace('$', '')
    # Remove \rm
    text = text.replace('\\rm ', '')
    text = text.replace('\\rm', '')
    
    # Replace common symbols
    text = text.replace('\\Delta ', 'Δ')
    text = text.replace('\\Delta', 'Δ')
    text = text.replace('\\parallel ', '∥')
    text = text.replace('\\parallel', '∥')
    text = text.replace('\\perp ', '⟂')
    text = text.replace('\\perp', '⟂')
    
    if is_latex:
        # Handle subscripts: _{...} or _a
        text = re.sub(r'_\{([^}]+)\}', r'<sub>\1</sub>', text)
        text = re.sub(r'_([a-zA-Z0-9])', r'<sub>\1</sub>', text)
        
        # Handle superscripts: ^{...} or ^a
        text = re.sub(r'\^\{([^}]+)\}', r'<sup>\1</sup>', text)
        text = re.sub(r'\^([a-zA-Z0-9])', r'<sup>\1</sup>', text)
    
    return text

print("1:", format_latex("mms3_fgm_b_lmn_srvy_l2_0"))
print("2:", format_latex("$\\Delta \\rm v_{\\rm i,L}$"))
print("3:", format_latex("$\\parallel$"))
print("4:", format_latex("$\\perp$"))

