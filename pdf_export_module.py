"""
PDF Export Module for Confined Polymer Analysis
Generates professional PDF reports with figures and statistics
"""

from io import BytesIO
from datetime import datetime
import numpy as np
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY


def export_pdf_report(fig_dict, results_dict, params_dict):
    """
    Generate PDF report from figures and results
    
    Parameters:
    -----------
    fig_dict : dict
        Dictionary containing matplotlib figures
    results_dict : dict
        Dictionary containing numerical results
    params_dict : dict
        Dictionary containing simulation parameters
    
    Returns:
    --------
    BytesIO
        PDF document as bytes buffer, or None if error
    """
    
    try:
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter,
                               rightMargin=0.75*inch,
                               leftMargin=0.75*inch,
                               topMargin=0.75*inch,
                               bottomMargin=0.75*inch)
        
        # Container for PDF elements
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2E86DE'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2E86DE'),
            spaceAfter=10,
            fontName='Helvetica-Bold'
        )
        
        # Title
        title = Paragraph("🧬 Confined Polymer Analysis Report", title_style)
        elements.append(title)
        elements.append(Spacer(1, 0.3*inch))
        
        # Timestamp
        timestamp_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        elements.append(Paragraph(timestamp_text, styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Parameters Section
        elements.append(Paragraph("Simulation Parameters", heading_style))
        
        params_data = [
            ['Parameter', 'Value'],
            ['Chain Length (N)', str(params_dict.get('N', 'N/A'))],
            ['Kuhn Length a (µm)', f"{params_dict.get('a', 0):.4f}"],
            ['Confinement Length L (µm)', f"{params_dict.get('L', 0):.4f}"],
            ['Cylinder Radius R (µm)', f"{params_dict.get('R', 0):.4f}"],
            ['Tethering Point x₀ (µm)', f"{params_dict.get('x0', 0):.4f}"],
            ['Persistence Length lₚ (µm)', f"{params_dict.get('lp', 0):.4f}"]
        ]
        
        params_table = Table(params_data, colWidths=[3*inch, 2*inch])
        params_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86DE')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(params_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Results Section
        elements.append(Paragraph("Simulation Results", heading_style))
        
        # Add summary statistics if available
        if 'fjc' in results_dict or 'saw' in results_dict or 'wlc' in results_dict:
            stats_data = [['Model', 'Samples', 'Mean (µm)', 'Std Dev (µm)']]
            
            if 'fjc' in results_dict:
                fjc_res = results_dict['fjc']
                stats_data.append([
                    'FJC',
                    str(fjc_res.get('samples', 'N/A')),
                    f"{fjc_res.get('mean', 0):.6f}",
                    f"{fjc_res.get('std', 0):.6f}"
                ])
            
            if 'saw' in results_dict:
                saw_res = results_dict['saw']
                stats_data.append([
                    'SAW',
                    str(saw_res.get('samples', 'N/A')),
                    f"{np.mean(saw_res.get('data', [0])):.6f}",
                    f"{np.std(saw_res.get('data', [0])):.6f}"
                ])
            
            if 'wlc' in results_dict:
                wlc_res = results_dict['wlc']
                stats_data.append([
                    'WLC',
                    str(wlc_res.get('samples', 'N/A')),
                    f"{wlc_res.get('mean', 0):.6f}",
                    f"{wlc_res.get('std', 0):.6f}"
                ])
            
            stats_table = Table(stats_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86DE')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(stats_table)
            elements.append(Spacer(1, 0.3*inch))
        
        # Figure section
        if 'px_figure' in fig_dict:
            elements.append(PageBreak())
            elements.append(Paragraph("Distribution Plot", heading_style))
            
            fig = fig_dict['px_figure']
            
            # Save figure to BytesIO
            img_buffer = BytesIO()
            fig.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
            img_buffer.seek(0)
            
            # Add image to PDF
            img = Image(img_buffer, width=6.5*inch, height=4*inch)
            elements.append(img)
            elements.append(Spacer(1, 0.2*inch))
            elements.append(Paragraph(
                "P(x) Distribution comparing different polymer models (FJC, SAW, WLC) with analytical solutions.",
                styles['Normal']
            ))
        
        # Footer
        elements.append(Spacer(1, 0.3*inch))
        footer_text = "Confined Polymer Analysis - EXACT Implementation | Generated by Streamlit"
        elements.append(Paragraph(footer_text, ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER
        )))
        
        # Build PDF
        doc.build(elements)
        pdf_buffer.seek(0)
        
        return pdf_buffer
        
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        return None
