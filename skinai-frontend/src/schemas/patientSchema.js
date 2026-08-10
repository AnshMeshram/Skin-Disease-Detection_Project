import { z } from 'zod';

export const patientInfoSchema = z.object({
  name: z.string().optional(),
  gender: z.enum(['Male', 'Female', 'Other', '']).optional(),
  age: z
    .union([z.number(), z.string()])
    .optional()
    .transform((val) => (val === '' || val === undefined ? undefined : Number(val)))
    .refine((val) => val === undefined || (!isNaN(val) && val >= 0 && val <= 120), {
      message: 'Age must be a valid number between 0 and 120',
    }),
  site: z.string().optional(),
});

export const validatePatientInfo = (data) => {
  const result = patientInfoSchema.safeParse(data);
  if (!result.success) {
    const formatted = result.error.format();
    return { isValid: false, errors: formatted };
  }
  return { isValid: true, data: result.data };
};
